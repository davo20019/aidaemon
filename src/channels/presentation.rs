//! Shared presentation rules for outbound chat text.
//!
//! Keep channel messages readable here instead of teaching every notification
//! producer how to lay out a mobile-friendly card.

use once_cell::sync::Lazy;
use regex::{Captures, Regex};

const MAX_INLINE_STEP_CHARS: usize = 180;
const MAX_SCHEDULED_FAILURE_CHARS: usize = 900;

static URL_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"https?://[^\s<>]+").expect("valid outbound URL regex"));
static LOCAL_PATH_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)(^|[\s(])((?:~/|/)[^\s,;]+)").expect("valid local path regex"));
static TASK_COUNT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)^\d+\s*/\s*\d+\s+tasks?\s+(?:completed|done)\.?$")
        .expect("valid task-count regex")
});

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
        "escalation"
        | "mandate_ask"
        | "mandate_reconciliation_required"
        | "mandate_objective_control_required" => "⚠️ **Action needed**",
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

/// Render a recurring scheduled-run result as a compact mobile card.
///
/// Executor results deliberately retain detailed paths, commands, versions, and
/// verification evidence in the durable task ledger. A chat notification has a
/// different job: state the outcome, link the deliverable, and make the schedule
/// state obvious. Keeping that boundary here prevents raw execution transcripts
/// from becoming the default Telegram experience.
pub(crate) fn present_scheduled_run_notification(
    notification_type: &str,
    message: &str,
    recurring: bool,
) -> String {
    let prepared = prepare_chat_message(message);
    let links = presentable_urls(&prepared);
    let schedule_note = recurring.then_some("_Recurring schedule remains active._");

    if notification_type == "completed" {
        let outcome = concise_scheduled_outcome(&prepared);
        let mut sections = vec!["✅ **Scheduled run complete**".to_string()];
        sections.push(if outcome.is_empty() {
            "Completed successfully.".to_string()
        } else {
            outcome
        });

        for (index, (url, host)) in links.iter().take(2).enumerate() {
            let label = if links.len() == 1 {
                format!("Open {host}")
            } else {
                format!("Open result {} · {host}", index + 1)
            };
            sections.push(format!("🔗 [{label}]({url})"));
        }
        sections.push("**Checks:** Passed".to_string());
        if let Some(schedule_note) = schedule_note {
            sections.push(schedule_note.to_string());
        }
        return sections.join("\n\n");
    }

    let heading = if notification_type == "failed" {
        "⚠️ **Scheduled run needs attention**"
    } else {
        "ℹ️ **Scheduled run update**"
    };
    let cleaned = shorten_local_paths(&prepared);
    let detail = truncate_chars(cleaned.trim(), MAX_SCHEDULED_FAILURE_CHARS);
    match (detail.is_empty(), schedule_note) {
        (true, Some(schedule_note)) => format!("{heading}\n\n{schedule_note}"),
        (true, None) => heading.to_string(),
        (false, Some(schedule_note)) => format!("{heading}\n\n{detail}\n\n{schedule_note}"),
        (false, None) => format!("{heading}\n\n{detail}"),
    }
}

fn presentable_urls(text: &str) -> Vec<(String, String)> {
    let mut urls = Vec::new();
    for found in URL_RE.find_iter(text) {
        let candidate = found
            .as_str()
            .trim_end_matches(['.', ',', ';', ':', ')', ']', '}']);
        let Ok(parsed) = reqwest::Url::parse(candidate) else {
            continue;
        };
        let Some(host) = parsed.host_str() else {
            continue;
        };
        let private_host = host.eq_ignore_ascii_case("localhost")
            || host == "127.0.0.1"
            || host == "::1"
            || host.ends_with(".local");
        if private_host || urls.iter().any(|(existing, _)| existing == candidate) {
            continue;
        }
        urls.push((candidate.to_string(), host.to_string()));
    }
    urls
}

fn concise_scheduled_outcome(text: &str) -> String {
    let without_urls = URL_RE.replace_all(text, "");
    let shortened = shorten_local_paths(&without_urls);
    for line in shortened.lines().map(str::trim) {
        if line.is_empty()
            || TASK_COUNT_RE.is_match(line)
            || (line.starts_with("**") && line.ends_with("**"))
        {
            continue;
        }
        let sentence = first_sentence(line);
        if !sentence.is_empty() {
            return truncate_chars(sentence, 320);
        }
    }
    String::new()
}

fn first_sentence(text: &str) -> &str {
    let bytes = text.as_bytes();
    for (index, byte) in bytes.iter().enumerate() {
        if matches!(byte, b'.' | b'!' | b'?')
            && bytes.get(index + 1).is_none_or(u8::is_ascii_whitespace)
        {
            return text[..=index].trim();
        }
    }
    text.trim()
}

fn shorten_local_paths(text: &str) -> String {
    LOCAL_PATH_RE
        .replace_all(text, |captures: &Captures<'_>| {
            let prefix = captures.get(1).map_or("", |m| m.as_str());
            let original = captures.get(2).map_or("", |m| m.as_str());
            let trimmed = original.trim_end_matches(['.', ')', ']', '}']);
            let suffix = &original[trimmed.len()..];
            let filename = std::path::Path::new(trimmed)
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| !name.is_empty())
                .unwrap_or("local file");
            format!("{prefix}`{filename}`{suffix}")
        })
        .into_owned()
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut truncated = text
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    truncated.push('…');
    truncated
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

    #[test]
    fn scheduled_completion_is_compact_and_hides_local_path() {
        let input =
            "Published /Users/synthetic/projects/blog/src/content/posts/2026-08-12-example.md. \
                     Evidence was repository guidance and clean git status. npm run deploy built \
                     successfully with Vite 7.3.1 and uploaded 5 assets. Exact public URL \
                     https://blog.example.com/posts/example/ returned HTTP 200.";

        let output = present_scheduled_run_notification("completed", input, true);

        assert!(output.starts_with("✅ **Scheduled run complete**"));
        assert!(output.contains("Published `2026-08-12-example.md`."));
        assert!(output.contains("[Open blog.example.com](https://blog.example.com/posts/example/)"));
        assert!(output.contains("**Checks:** Passed"));
        assert!(output.contains("_Recurring schedule remains active._"));
        assert!(!output.contains("/Users/synthetic"));
        assert!(!output.contains("Vite 7.3.1"));
        assert!(!output.contains("uploaded 5 assets"));
    }

    #[test]
    fn scheduled_completion_skips_task_count_and_internal_urls() {
        let input = "2/2 tasks completed.\n\n**Publish the article**\nPublished the article. \
                     Previewed at http://localhost:4173/ and deployed to \
                     https://blog.example.com/posts/article/.";

        let output = present_scheduled_run_notification("completed", input, true);

        assert!(output.contains("Published the article."));
        assert!(output.contains("https://blog.example.com/posts/article/"));
        assert!(!output.contains("localhost"));
        assert!(!output.contains("2/2 tasks"));
    }

    #[test]
    fn scheduled_failure_keeps_actionable_detail_but_shortens_paths() {
        let input = "Deployment failed while reading /Users/synthetic/projects/blog/wrangler.toml.\n\nNeeded: refresh the deployment token.";

        let output = present_scheduled_run_notification("failed", input, true);

        assert!(output.starts_with("⚠️ **Scheduled run needs attention**"));
        assert!(output.contains("`wrangler.toml`"));
        assert!(output.contains("refresh the deployment token"));
        assert!(!output.contains("/Users/synthetic"));
        assert!(output.contains("_Recurring schedule remains active._"));
    }
}
