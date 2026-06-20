//! Spill oversized tool results to disk so the model can recover the full data
//! via read_file paging / terminal jq/grep, instead of losing the middle to
//! in-context head+tail compression. The on-disk file is the full, untruncated
//! result; the returned preview is a bounded head + structural summary + an
//! anti-fabrication pointer to the file.
//!
//! The public API (`build_spilled_preview`, `spill_dir`) is wired into the
//! agent loop in a later commit; `#![allow(dead_code)]` suppresses spurious
//! "never used" lints until that wiring lands.
#![allow(dead_code)]

use std::path::PathBuf;

/// Base directory for spilled tool results, under the OS temp dir
/// (`std::env::temp_dir()` honors `TMPDIR`/`%TEMP%`, cross-platform). Spilled
/// results are ephemeral scratch read within the session, not persistent state,
/// so they live in temp rather than `~/.aidaemon/`. Returns `Some` always, but
/// kept as `Option` so callers uniformly fall back to lossy compression on any
/// future failure mode.
pub fn spill_dir() -> Option<PathBuf> {
    Some(std::env::temp_dir().join("aidaemon").join("tool_results"))
}

/// Characters reserved from the per-model cap for the summary + notice so the
/// preview stays close to the cap rather than overshooting it badly.
const SPILL_ANNOTATION_RESERVE: usize = 512;

/// Filesystem-safe session id (e.g. `telegram:12345` -> `telegram_12345`).
fn sanitize_session_id(session_id: &str) -> String {
    session_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Public entry: spill under `<temp_dir>/aidaemon/tool_results` where
/// `<temp_dir>` is `std::env::temp_dir()` (honors `TMPDIR`/`%TEMP%`).
pub fn build_spilled_preview(
    tool_name: &str,
    session_id: &str,
    full_text: &str,
    max_chars: usize,
) -> Option<String> {
    build_spilled_preview_in(spill_dir()?, tool_name, session_id, full_text, max_chars)
}

/// Testable core: spill under an explicit base directory.
fn build_spilled_preview_in(
    base_dir: PathBuf,
    tool_name: &str,
    session_id: &str,
    full_text: &str,
    max_chars: usize,
) -> Option<String> {
    let dir = base_dir.join(sanitize_session_id(session_id));
    std::fs::create_dir_all(&dir).ok()?;

    // Pretty-print JSON so line reads + grep on field names work; else verbatim.
    let parsed: Option<serde_json::Value> = serde_json::from_str(full_text).ok();
    let (stored_text, ext) = match &parsed {
        Some(v) => (
            serde_json::to_string_pretty(v).unwrap_or_else(|_| full_text.to_string()),
            "json",
        ),
        None => (full_text.to_string(), "txt"),
    };

    let short_id: String = uuid::Uuid::new_v4()
        .simple()
        .to_string()
        .chars()
        .take(8)
        .collect();
    let path = dir.join(format!("{}-{}.{}", tool_name, short_id, ext));
    std::fs::write(&path, stored_text.as_bytes()).ok()?;
    let abs_path = path.to_string_lossy().into_owned();

    let total_chars = stored_text.chars().count();
    let head_chars = max_chars
        .saturating_sub(SPILL_ANNOTATION_RESERVE)
        .max(256)
        .min(max_chars);
    let head: String = stored_text.chars().take(head_chars).collect();
    let shown_chars = head.chars().count();

    let summary = parsed
        .as_ref()
        .map(json_structure_summary)
        .unwrap_or_default();
    let summary_block = if summary.is_empty() {
        String::new()
    } else {
        format!("{}\n\n", summary)
    };

    Some(format!(
        "{head}\n\n{summary_block}{notice}",
        head = head,
        summary_block = summary_block,
        notice = spill_notice(shown_chars, total_chars, &abs_path),
    ))
}

/// One-line structural summary so the model knows what to `jq`/`grep` for.
fn json_structure_summary(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(map) => {
            let keys: Vec<&str> = map.keys().map(String::as_str).collect();
            format!("[JSON object — top-level keys: {}]", keys.join(", "))
        }
        serde_json::Value::Array(items) => {
            let sample_keys = items
                .first()
                .and_then(|i| i.as_object())
                .map(|m| m.keys().map(String::as_str).collect::<Vec<_>>().join(", "))
                .unwrap_or_default();
            if sample_keys.is_empty() {
                format!("[JSON array — {} items]", items.len())
            } else {
                format!(
                    "[JSON array — {} items; item keys: {}]",
                    items.len(),
                    sample_keys
                )
            }
        }
        _ => String::new(),
    }
}

fn spill_notice(shown_chars: usize, total_chars: usize, abs_path: &str) -> String {
    let omitted = total_chars.saturating_sub(shown_chars);
    format!(
        "[⚠ LARGE RESULT — full {total} chars saved to {path}. Only the first {shown} chars are \
         shown above; {omitted} chars are NOT visible to you here. To get the rest: read_file with \
         start_line/end_line to page through it, or query via terminal (e.g. `grep -n <term> {path}`, \
         `wc -l {path}`, or `jq '<path.into.structure>' {path}` for JSON). Do NOT enumerate, list, \
         count, or quote items that are not literally shown above — inventing the omitted content is \
         an error.]",
        total = total_chars,
        path = abs_path,
        shown = shown_chars,
        omitted = omitted,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_result_spills_full_body_and_preview_points_to_file() {
        let dir = tempfile::tempdir().unwrap();
        // Synthetic clinical-trial-style payload; "Fairfax" sits deep enough to be
        // dropped by head+tail compression but must survive in the spilled file.
        let mut locations = String::new();
        for i in 0..200 {
            locations.push_str(&format!(
                "{{\"city\":\"City{}\",\"facility\":\"Center {}\"}},",
                i, i
            ));
        }
        let full = format!(
            "{{\"locations\":[{}{{\"city\":\"Fairfax\",\"facility\":\"Synthetic Cancer Center\"}}]}}",
            locations
        );

        let preview = build_spilled_preview_in(
            dir.path().to_path_buf(),
            "http_request",
            "telegram:12345",
            &full,
            400,
        )
        .expect("spill should succeed");

        // Preview is bounded and honest.
        assert!(preview.contains("LARGE RESULT"));
        assert!(preview.contains("Do NOT enumerate"));
        assert!(preview.contains("[JSON object"));
        // Preview references the on-disk file by absolute path.
        let session_dir = dir.path().join("telegram_12345");
        let written: Vec<_> = std::fs::read_dir(&session_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .collect();
        assert_eq!(written.len(), 1);
        let path = &written[0];
        assert_eq!(path.extension().unwrap(), "json");
        assert!(preview.contains(&path.to_string_lossy().into_owned()));
        // The full body — including the would-be-dropped middle — is on disk.
        let on_disk = std::fs::read_to_string(path).unwrap();
        assert!(on_disk.contains("Fairfax"));
        // ...but NOT in the bounded preview.
        assert!(!preview.contains("Fairfax"));
    }

    #[test]
    fn preview_head_does_not_exceed_small_cap() {
        let dir = tempfile::tempdir().unwrap();
        let full = "x".repeat(10_000);
        let preview =
            build_spilled_preview_in(dir.path().to_path_buf(), "terminal", "s:1", &full, 300)
                .unwrap();
        // The head is everything before the notice block; it must not exceed the cap.
        let head = preview.split("[⚠ LARGE RESULT").next().unwrap();
        assert!(
            head.chars().count() <= 300,
            "head was {} chars",
            head.chars().count()
        );
    }

    #[test]
    fn non_json_result_spills_as_txt() {
        let dir = tempfile::tempdir().unwrap();
        let full = "line\n".repeat(5000);
        let preview =
            build_spilled_preview_in(dir.path().to_path_buf(), "terminal", "slack:U1", &full, 300)
                .expect("spill should succeed");
        let session_dir = dir.path().join("slack_U1");
        let path = std::fs::read_dir(&session_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .next()
            .unwrap()
            .path();
        assert_eq!(path.extension().unwrap(), "txt");
        assert!(preview.contains("LARGE RESULT"));
        // No JSON summary for plain text.
        assert!(!preview.contains("[JSON"));
    }
}
