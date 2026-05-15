use crate::traits::SpecialistKind;
use serde::Deserialize;
use tracing::warn;

use super::{SpecialistDef, SpecialistSource};

#[allow(dead_code)] // fields populated by serde; consumed by loader in upcoming tasks
#[derive(Debug, Deserialize)]
struct RawFrontmatter {
    kind: String,
    description: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    tools: Option<Vec<String>>,
    #[serde(default)]
    max_iterations: Option<usize>,
    #[serde(default)]
    tool_budget: Option<usize>,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(flatten)]
    extra: std::collections::BTreeMap<String, serde_yaml::Value>,
}

#[allow(dead_code)] // consumed by registry loader in upcoming tasks
pub fn parse_specialist(
    expected_kind: SpecialistKind,
    content: &str,
) -> anyhow::Result<SpecialistDef> {
    let (frontmatter_str, body) = split_frontmatter(content)?;
    let raw: RawFrontmatter = serde_yaml::from_str(&frontmatter_str)
        .map_err(|e| anyhow::anyhow!("invalid frontmatter YAML: {}", e))?;

    let declared_kind = SpecialistKind::from_str(&raw.kind)
        .ok_or_else(|| anyhow::anyhow!("unknown kind in frontmatter: {}", raw.kind))?;
    if declared_kind != expected_kind {
        anyhow::bail!(
            "kind mismatch: file declares {} but expected {}",
            raw.kind,
            expected_kind.as_str()
        );
    }

    let body_trimmed = body.trim();
    if body_trimmed.is_empty() {
        anyhow::bail!("specialist body is empty");
    }

    if !raw.extra.is_empty() {
        for key in raw.extra.keys() {
            warn!(kind = %expected_kind.as_str(), key = %key, "unknown specialist frontmatter key — ignored");
        }
    }

    Ok(SpecialistDef {
        kind: expected_kind,
        description: raw.description,
        system_prompt_template: body,
        model: raw.model,
        tools: raw.tools,
        max_iterations: raw.max_iterations,
        tool_budget: raw.tool_budget,
        timeout_secs: raw.timeout_secs,
        source: SpecialistSource::Bundled, // default; overridden by loader in Task 3
    })
}

fn split_frontmatter(content: &str) -> anyhow::Result<(String, String)> {
    let content = content.trim_start_matches('\u{feff}');
    // Normalize CRLF -> LF once so the rest of the logic deals with a single
    // line-ending convention. Per spec, both endings must be tolerated.
    let normalized: String = if content.contains("\r\n") {
        content.replace("\r\n", "\n")
    } else {
        content.to_string()
    };
    let rest = normalized
        .strip_prefix("---\n")
        .ok_or_else(|| anyhow::anyhow!("missing opening `---\\n` frontmatter delimiter"))?;
    let (frontmatter, body) = rest
        .split_once("\n---\n")
        .ok_or_else(|| anyhow::anyhow!("missing closing `---` frontmatter delimiter"))?;
    Ok((frontmatter.to_string(), body.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::SpecialistKind;

    #[test]
    fn parses_minimal_valid_file() {
        let content =
            "---\nkind: code\ndescription: Code specialist.\n---\n\nYou are a code specialist.\n";
        let def = parse_specialist(SpecialistKind::Code, content).expect("parse ok");
        assert_eq!(def.kind, SpecialistKind::Code);
        assert_eq!(def.description, "Code specialist.");
        assert_eq!(
            def.system_prompt_template.trim(),
            "You are a code specialist."
        );
        assert_eq!(def.model, None);
        assert_eq!(def.tools, None);
        assert_eq!(def.max_iterations, None);
        assert_eq!(def.tool_budget, None);
        assert_eq!(def.timeout_secs, None);
    }

    #[test]
    fn parses_full_frontmatter() {
        let content = "---\n\
kind: code\n\
description: Code specialist.\n\
model: claude-sonnet-4-6\n\
tools:\n  - read_file\n  - write_file\n\
max_iterations: 20\n\
tool_budget: 35\n\
timeout_secs: 600\n\
---\n\nBody here.\n";
        let def = parse_specialist(SpecialistKind::Code, content).expect("parse ok");
        assert_eq!(def.model.as_deref(), Some("claude-sonnet-4-6"));
        assert_eq!(
            def.tools.as_deref(),
            Some(&["read_file".to_string(), "write_file".to_string()][..])
        );
        assert_eq!(def.max_iterations, Some(20));
        assert_eq!(def.tool_budget, Some(35));
        assert_eq!(def.timeout_secs, Some(600));
    }

    #[test]
    fn rejects_kind_mismatch() {
        let content = "---\nkind: research\ndescription: Mismatched.\n---\n\nBody.\n";
        assert!(parse_specialist(SpecialistKind::Code, content).is_err());
    }

    #[test]
    fn rejects_missing_frontmatter_separator() {
        assert!(parse_specialist(SpecialistKind::Code, "no frontmatter").is_err());
    }

    #[test]
    fn rejects_empty_body() {
        let content = "---\nkind: code\ndescription: d.\n---\n   \n";
        assert!(parse_specialist(SpecialistKind::Code, content).is_err());
    }

    #[test]
    fn unknown_frontmatter_keys_are_ignored() {
        let content = "---\nkind: code\ndescription: d.\nfuture_field: nope\n---\nBody.\n";
        assert!(parse_specialist(SpecialistKind::Code, content).is_ok());
    }

    #[test]
    fn tolerates_crlf_line_endings() {
        let content = "---\r\nkind: code\r\ndescription: CRLF specialist.\r\n---\r\n\r\nBody for {{mission}}.\r\n";
        let def = parse_specialist(SpecialistKind::Code, content).expect("CRLF should parse");
        assert_eq!(def.kind, SpecialistKind::Code);
        assert_eq!(def.description, "CRLF specialist.");
        // After CRLF normalization the body retains LF line endings.
        assert!(def.system_prompt_template.contains("Body for {{mission}}."));
        assert!(!def.system_prompt_template.contains('\r'));
    }
}
