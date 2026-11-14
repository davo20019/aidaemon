//! The daemon's own markers, rendered for the owner.
//!
//! Refusal reasons, redaction placeholders and the daemon's file inbox are
//! written for the model: it reads `[OWNER DECLINED]` or
//! `[REDACTED:File path]` and acts on them. The owner should never see them,
//! so text leaving for a chat goes through [`render_for_owner`], which
//! replaces each kind it finds with plain words. The list is the daemon's
//! own marker constants, not phrases: a marker it does not define is left
//! alone.

use std::sync::RwLock;

use regex::Regex;
use std::sync::LazyLock as Lazy;

use crate::sanitize::{ENDPOINT_LABEL, FILE_PATH_LABEL, REDACTED_MARKER, WINDOWS_PATH_LABEL};

/// One kind of internal marker the owner must not see.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InternalMarker {
    /// [`aidaemon_core::OWNER_DECLINED`]: the owner declined a call.
    OwnerDeclined,
    /// [`aidaemon_core::EGRESS_BLOCKED`]: an outbound call carried a secret.
    EgressBlocked,
    /// [`aidaemon_core::AUTHORITY_BLOCKED`]: the call was outside the run's
    /// authority.
    AuthorityBlocked,
    /// [`aidaemon_core::UNTRUSTED_BLOCKED`]: the safety check held a call
    /// back after the run read outside content.
    UntrustedBlocked,
    /// A redaction placeholder (`[REDACTED]`, `[REDACTED:<label>]`).
    Redaction,
    /// A path inside the daemon's file inbox.
    InboxPath,
}

/// The fixed-token markers and what the owner reads instead.
const TOKEN_MARKERS: [(InternalMarker, &str, &str); 4] = [
    (
        InternalMarker::OwnerDeclined,
        aidaemon_core::OWNER_DECLINED,
        "you declined that",
    ),
    (
        InternalMarker::EgressBlocked,
        aidaemon_core::EGRESS_BLOCKED,
        "blocked because it would have sent a secret",
    ),
    (
        InternalMarker::AuthorityBlocked,
        aidaemon_core::AUTHORITY_BLOCKED,
        "not allowed in this chat",
    ),
    (
        InternalMarker::UntrustedBlocked,
        aidaemon_core::UNTRUSTED_BLOCKED,
        "held back by a safety check",
    ),
];

/// The daemon's file inbox when the configuration does not move it.
pub const DEFAULT_INBOX_DIR: &str = "~/.aidaemon/files/inbox";

static INBOX_DIRS: Lazy<RwLock<Vec<String>>> = Lazy::new(|| {
    let mut known = Vec::new();
    push_dir_forms(&mut known, DEFAULT_INBOX_DIR);
    RwLock::new(known)
});

/// Add `dir`, `~/...` and expanded forms, to the dirs that are rendered.
fn push_dir_forms(known: &mut Vec<String>, dir: &str) {
    let dir = dir.trim().trim_end_matches('/');
    if dir.is_empty() {
        return;
    }
    let mut forms = vec![dir.to_string()];
    if let Some(home) = dirs::home_dir() {
        let home = home.to_string_lossy().trim_end_matches('/').to_string();
        if let Some(rest) = dir.strip_prefix("~/") {
            forms.push(format!("{home}/{rest}"));
        } else if let Some(rest) = dir.strip_prefix(&format!("{home}/")) {
            forms.push(format!("~/{rest}"));
        }
    }
    for form in forms {
        if form.len() > 2 && !known.contains(&form) {
            known.push(form);
        }
    }
}

/// The configured file inbox, so paths in it are rendered too.
pub fn register_inbox_dir(dir: &str) {
    if let Ok(mut known) = INBOX_DIRS.write() {
        push_dir_forms(&mut known, dir);
    }
}

static REDACTION: Lazy<Regex> = Lazy::new(|| {
    Regex::new(&format!(
        r"{}(?::([^\]\n]{{1,40}}))?\]",
        regex::escape(REDACTED_MARKER)
    ))
    .expect("valid redaction regex")
});

/// What the owner reads for a redaction labelled `label`.
fn redaction_words(label: Option<&str>) -> &'static str {
    match label {
        Some(FILE_PATH_LABEL) | Some(WINDOWS_PATH_LABEL) => "a file",
        Some(ENDPOINT_LABEL) => "an address",
        _ => "a hidden value",
    }
}

/// The name the owner gave a file the inbox saved as `<8 hex>_<name>`.
fn inbox_file_name(stored: &str) -> &str {
    match stored.split_once('_') {
        Some((prefix, name))
            if prefix.len() == 8
                && prefix.chars().all(|c| c.is_ascii_hexdigit())
                && !name.is_empty() =>
        {
            name
        }
        _ => stored,
    }
}

/// Replace inbox paths in `text` with the file's own name.
fn render_inbox_paths(text: &str) -> Option<String> {
    let mut known = INBOX_DIRS.read().ok()?.clone();
    let mut out = text.to_string();
    let mut changed = false;
    // Longest first, so `/home/x/.aidaemon/...` wins over a shorter form.
    known.sort_by_key(|dir| std::cmp::Reverse(dir.len()));
    for dir in known {
        let prefix = format!("{dir}/");
        while let Some(at) = out.find(&prefix) {
            let rest = &out[at + prefix.len()..];
            let end = rest
                .find(|c: char| c.is_whitespace() || "`'\")]>,;".contains(c))
                .unwrap_or(rest.len());
            let mut stored = &rest[..end];
            stored = stored.trim_end_matches(['.', ':', '!', '?']);
            let consumed = stored.len();
            let shown = if stored.is_empty() || stored.contains('/') {
                "a file you sent".to_string()
            } else {
                inbox_file_name(stored).to_string()
            };
            out.replace_range(at..at + prefix.len() + consumed, &shown);
            changed = true;
        }
    }
    changed.then_some(out)
}

/// `text` with every internal marker replaced by owner-facing words, and
/// the kinds that were found (empty when nothing changed).
pub fn render_for_owner(text: &str) -> (String, Vec<InternalMarker>) {
    let mut out = text.to_string();
    let mut found = Vec::new();
    for (kind, token, words) in TOKEN_MARKERS {
        if out.contains(token) {
            out = out.replace(token, words);
            found.push(kind);
        }
    }
    if REDACTION.is_match(&out) {
        out = REDACTION
            .replace_all(&out, |caps: &regex::Captures<'_>| {
                redaction_words(caps.get(1).map(|m| m.as_str()))
            })
            .into_owned();
        found.push(InternalMarker::Redaction);
    }
    if let Some(rendered) = render_inbox_paths(&out) {
        out = rendered;
        found.push(InternalMarker::InboxPath);
    }
    (out, found)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_marker_constant_becomes_plain_words() {
        for (kind, token, words) in TOKEN_MARKERS {
            let (out, found) = render_for_owner(&format!("The upload stopped: {token} sorry."));
            assert_eq!(found, vec![kind]);
            assert_eq!(out, format!("The upload stopped: {words} sorry."));
        }
    }

    #[test]
    fn redactions_say_what_was_hidden_without_the_placeholder() {
        let text = format!(
            "Copied {REDACTED_MARKER}:{FILE_PATH_LABEL}] to {REDACTED_MARKER}:{ENDPOINT_LABEL}] \
             with {REDACTED_MARKER}:API key] and {REDACTED_MARKER}]."
        );
        let (out, found) = render_for_owner(&text);
        assert_eq!(found, vec![InternalMarker::Redaction]);
        assert_eq!(
            out,
            "Copied a file to an address with a hidden value and a hidden value."
        );
        // What the redactor writes is what this renders.
        let redacted = crate::sanitize::redact_secrets("open /Users/alice/projects/site/a.png");
        let (out, _) = render_for_owner(&redacted);
        assert_eq!(out, "open a file");
    }

    #[test]
    fn an_inbox_path_becomes_the_file_name_the_owner_sent() {
        let (out, found) = render_for_owner(
            "Saved it from ~/.aidaemon/files/inbox/dd4dc47d_screenshot.png, then resized it.",
        );
        assert_eq!(found, vec![InternalMarker::InboxPath]);
        assert_eq!(out, "Saved it from screenshot.png, then resized it.");

        let (out, _) = render_for_owner("`~/.aidaemon/files/inbox/report.pdf`");
        assert_eq!(out, "`report.pdf`");
    }

    #[test]
    fn a_configured_inbox_is_rendered_too() {
        register_inbox_dir("/srv/synthetic-inbox/");
        let (out, found) = render_for_owner("See /srv/synthetic-inbox/0a1b2c3d_notes.txt.");
        assert_eq!(found, vec![InternalMarker::InboxPath]);
        assert_eq!(out, "See notes.txt.");
    }

    #[test]
    fn ordinary_text_is_untouched() {
        let text = "Fixed issue #12 in ~/projects/site/index.php; [OWNER] is a label.";
        let (out, found) = render_for_owner(text);
        assert!(found.is_empty());
        assert_eq!(out, text);
    }
}
