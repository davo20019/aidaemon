//! Reading a reviewer's output into a structured verdict.

use crate::*;

pub(crate) fn truncate_with_note(value: &str, max_chars: usize) -> String {
    let clipped: String = value.chars().take(max_chars).collect();
    if clipped.chars().count() < value.chars().count() {
        format!(
            "{}\n[... truncated to {} chars ...]",
            clipped.trim_end(),
            max_chars
        )
    } else {
        clipped
    }
}

pub(crate) fn extract_json_content(raw: &str) -> Option<String> {
    let v: Value = serde_json::from_str(raw).ok()?;
    if let Some(result) = v.get("result").and_then(|r| r.as_str()) {
        return Some(result.to_string());
    }
    if let Some(output) = v.get("output").and_then(|o| o.as_str()) {
        return Some(output.to_string());
    }
    if let Some(content) = v.get("content").and_then(|c| c.as_str()) {
        return Some(content.to_string());
    }
    if let Some(message) = v.get("message").and_then(|m| m.as_str()) {
        return Some(message.to_string());
    }
    None
}

pub(crate) fn extract_jsonl_content(raw: &str) -> Option<String> {
    let mut last_content: Option<String> = None;
    for line in raw.lines().rev() {
        if let Ok(v) = serde_json::from_str::<Value>(line)
            && let Some(content) = v
                .pointer("/item/content")
                .or_else(|| v.pointer("/content"))
                .or_else(|| v.pointer("/result"))
            && let Some(text) = content.as_str()
        {
            last_content = Some(text.to_string());
            break;
        }
    }
    last_content
}

pub(crate) fn extract_meaningful_cli_output(raw: &str, max_chars: usize) -> String {
    if let Some(content) = extract_json_content(raw) {
        return truncate_with_note(&content, max_chars);
    }
    if let Some(content) = extract_jsonl_content(raw) {
        return truncate_with_note(&content, max_chars);
    }
    truncate_with_note(raw, max_chars)
}

pub(crate) fn parse_review_candidate(text: &str) -> Option<ReviewResponseCandidate> {
    serde_json::from_str::<ReviewResponseCandidate>(text)
        .ok()
        .filter(|candidate| candidate.has_review_signal())
}

pub(crate) fn extract_review_candidate_from_json_objects(
    raw: &str,
) -> Option<ReviewResponseCandidate> {
    let bytes = raw.as_bytes();
    let mut depth = 0usize;
    let mut start: Option<usize> = None;
    let mut in_string = false;
    let mut escaped = false;
    for (idx, b) in bytes.iter().enumerate() {
        let ch = *b as char;
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
            continue;
        }
        if ch == '{' {
            if start.is_none() {
                start = Some(idx);
            }
            depth = depth.saturating_add(1);
            continue;
        }
        if ch == '}' && depth > 0 {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                if let Some(s) = start
                    && let Some(slice) = raw.get(s..=idx)
                {
                    if let Some(candidate) = parse_review_candidate(slice) {
                        return Some(candidate);
                    }
                    if let Ok(value) = serde_json::from_str::<Value>(slice) {
                        for pointer in [
                            "/result",
                            "/output",
                            "/content",
                            "/message",
                            "/item/content",
                        ] {
                            if let Some(text) = value.pointer(pointer).and_then(Value::as_str)
                                && let Some(candidate) = parse_review_candidate(text)
                            {
                                return Some(candidate);
                            }
                        }
                    }
                }
                start = None;
            }
        }
    }
    None
}

pub(crate) fn normalize_review_items(values: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for value in values {
        let cleaned = value.split_whitespace().collect::<Vec<_>>().join(" ");
        let trimmed = cleaned.trim();
        if trimmed.is_empty() {
            continue;
        }
        let clipped: String = trimmed.chars().take(REVIEW_LIST_ITEM_MAX_CHARS).collect();
        if seen.insert(clipped.clone()) {
            out.push(clipped);
        }
        if out.len() >= REVIEW_LIST_MAX_ITEMS {
            break;
        }
    }
    out
}

pub(crate) fn normalize_review_severity(value: Option<&str>) -> &'static str {
    let raw = value.unwrap_or("").trim().to_ascii_lowercase();
    match raw.as_str() {
        "critical" | "blocker" | "p0" => "critical",
        "high" | "major" | "p1" => "high",
        "medium" | "moderate" | "p2" => "medium",
        "low" | "minor" | "nit" | "p3" => "low",
        _ => "medium",
    }
}

pub(crate) fn normalize_review_line(value: Option<&Value>) -> Option<u64> {
    let raw = value?;
    if let Some(num) = raw.as_u64() {
        return (num > 0 && num < 1_000_000).then_some(num);
    }
    let parsed = raw.as_str()?.trim().parse::<u64>().ok()?;
    (parsed > 0 && parsed < 1_000_000).then_some(parsed)
}

pub(crate) fn normalize_review_findings(values: Vec<ReviewFindingCandidate>) -> Vec<Value> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for value in values {
        let issue = value
            .issue
            .unwrap_or_default()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let issue = issue.trim();
        if issue.is_empty() {
            continue;
        }
        let issue: String = issue.chars().take(REVIEW_LIST_ITEM_MAX_CHARS).collect();
        let fix = value
            .fix
            .unwrap_or_default()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let fix = fix.trim();
        let fix = if fix.is_empty() {
            None
        } else {
            Some(
                fix.chars()
                    .take(REVIEW_LIST_ITEM_MAX_CHARS)
                    .collect::<String>(),
            )
        };
        let file = value
            .file
            .unwrap_or_default()
            .trim()
            .chars()
            .take(300)
            .collect::<String>();
        let file = if file.is_empty() { None } else { Some(file) };
        let line = normalize_review_line(value.line.as_ref());
        let severity = normalize_review_severity(value.severity.as_deref());
        let dedupe_key = format!(
            "{}|{}|{}|{}",
            severity,
            file.clone().unwrap_or_default(),
            line.map(|v| v.to_string()).unwrap_or_default(),
            issue
        );
        if !seen.insert(dedupe_key) {
            continue;
        }
        out.push(json!({
            "severity": severity,
            "file": file,
            "line": line,
            "issue": issue,
            "fix": fix,
        }));
        if out.len() >= REVIEW_LIST_MAX_ITEMS {
            break;
        }
    }
    out
}

pub(crate) fn parse_review_output_payload(
    reviewer: &str,
    request_id: &str,
    scope: ReviewScope,
    diff_base: ReviewDiffBase,
    raw_output: &str,
) -> Result<Value, String> {
    let meaningful = extract_meaningful_cli_output(raw_output, REVIEW_MAX_OUTPUT_CHARS);
    let candidate = parse_review_candidate(&meaningful)
        .or_else(|| extract_review_candidate_from_json_objects(&meaningful))
        .or_else(|| extract_review_candidate_from_json_objects(raw_output));

    let Some(candidate) = candidate else {
        let excerpt = truncate_with_note(meaningful.trim(), REVIEW_SUMMARY_MAX_CHARS);
        let message = if excerpt.trim().is_empty() {
            "Reviewer returned output, but no structured JSON review payload was found.".to_string()
        } else {
            format!(
                "Reviewer returned output, but no structured JSON review payload was found.\n{}",
                excerpt
            )
        };
        return Err(message);
    };

    let requested_verdict = ReviewVerdict::parse(candidate.verdict.as_deref());
    let blocking = normalize_review_items(candidate.blocking.unwrap_or_default());
    let risks = normalize_review_items(candidate.risks.unwrap_or_default());
    let suggestions = normalize_review_items(candidate.suggestions.unwrap_or_default());
    let findings = normalize_review_findings(candidate.findings.unwrap_or_default());
    let has_non_low_findings = findings.iter().any(|finding| {
        matches!(
            finding.get("severity").and_then(Value::as_str),
            Some("critical" | "high" | "medium")
        )
    });
    // The reviewer's own verdict, read by exact value, can only be
    // tightened by its lists: an approve that names blocking items, risks
    // or a medium-or-worse finding is not an approve.
    let verdict = match requested_verdict {
        ReviewVerdict::Approve
            if blocking.is_empty() && risks.is_empty() && !has_non_low_findings =>
        {
            ReviewVerdict::Approve
        }
        _ => ReviewVerdict::NeedsChanges,
    };

    let mut summary = candidate.summary.unwrap_or_default();
    if summary.trim().is_empty() {
        summary = format!(
            "Structured review parsed (findings: {}, blocking: {}, risks: {}, suggestions: {}).",
            findings.len(),
            blocking.len(),
            risks.len(),
            suggestions.len()
        );
    }
    summary = truncate_with_note(summary.trim(), REVIEW_SUMMARY_MAX_CHARS);

    Ok(json!({
        "kind":"review_result",
        "request_id": request_id,
        "reviewer": reviewer,
        "scope": scope.as_str(),
        "diff_base": diff_base.as_str(),
        "verdict": verdict.as_str(),
        "findings": findings,
        "blocking": blocking,
        "risks": risks,
        "suggestions": suggestions,
        "summary": summary,
    }))
}
