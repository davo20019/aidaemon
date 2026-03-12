use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::agent::{tool_is_side_effecting, LearningContext};
use crate::tools::fs_utils;
use crate::traits::{ToolCallSemantics, ToolCapabilities, ToolTargetHint, ToolTargetHintKind};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EvidenceKind {
    FileRead,
    CommandOutput,
    GitState,
    ProcessState,
    ApiResponse,
    VerificationResult,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EvidenceTrust {
    Direct,
    Inferred,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidenceRecord {
    pub kind: EvidenceKind,
    pub source: String,
    pub observed_at: DateTime<Utc>,
    pub trust: EvidenceTrust,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<ToolTargetHint>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidenceState {
    pub target: Option<ToolTargetHint>,
    #[serde(default)]
    pub records: Vec<EvidenceRecord>,
    #[serde(default)]
    pub contradictions: Vec<String>,
    #[serde(default)]
    pub post_change_verification_done: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvidenceGateViolation {
    pub kind: EvidenceKind,
    pub reason: String,
    pub coaching: String,
    pub target: Option<String>,
}

impl EvidenceState {
    pub fn record_direct(
        &mut self,
        kind: EvidenceKind,
        source: impl Into<String>,
        targets: Vec<ToolTargetHint>,
    ) {
        if self.target.is_none() {
            self.target = targets.first().cloned();
        }
        self.records.push(EvidenceRecord {
            kind,
            source: source.into(),
            observed_at: Utc::now(),
            trust: EvidenceTrust::Direct,
            targets,
        });
    }

    fn has_direct_targeted_evidence(&self, kind: EvidenceKind, expected: &ToolTargetHint) -> bool {
        self.records.iter().any(|record| {
            record.kind == kind
                && record.trust == EvidenceTrust::Direct
                && record
                    .targets
                    .iter()
                    .any(|target| target_matches_expected(target, expected))
        })
    }
}

pub fn has_completed_side_effecting_tool_call(
    learning_ctx: &LearningContext,
    available_capabilities: &HashMap<String, ToolCapabilities>,
) -> bool {
    learning_ctx.tool_calls.iter().any(|tool_call| {
        tool_call
            .split('(')
            .next()
            .is_some_and(|name| tool_is_side_effecting(name, available_capabilities))
    })
}

pub fn assess_pre_execution_evidence_gate(
    tool_name: &str,
    raw_arguments: &str,
    evidence_state: &EvidenceState,
) -> Option<EvidenceGateViolation> {
    match tool_name {
        "edit_file" => {
            let target = path_target_hint(raw_arguments)?;
            if evidence_state.has_direct_targeted_evidence(EvidenceKind::FileRead, &target) {
                None
            } else {
                Some(EvidenceGateViolation {
                    kind: EvidenceKind::FileRead,
                    reason: "edit_file requires direct file-read evidence for the target path"
                        .to_string(),
                    coaching: format!(
                        "Before editing {}, use `read_file` on that exact path, then retry `edit_file`.",
                        target.value
                    ),
                    target: Some(target.value),
                })
            }
        }
        "write_file" => {
            let target = path_target_hint(raw_arguments)?;
            let path_exists = fs_utils::validate_path(&target.value)
                .map(|path| path.exists())
                .unwrap_or(false);
            if !path_exists
                || evidence_state.has_direct_targeted_evidence(EvidenceKind::FileRead, &target)
            {
                None
            } else {
                Some(EvidenceGateViolation {
                    kind: EvidenceKind::FileRead,
                    reason:
                        "overwriting an existing file requires direct file-read evidence first"
                            .to_string(),
                    coaching: format!(
                        "Before overwriting {}, use `read_file` on that path, then retry `write_file`.",
                        target.value
                    ),
                    target: Some(target.value),
                })
            }
        }
        "git_commit" => {
            let target = git_scope_target_hint(raw_arguments);
            if evidence_state.has_direct_targeted_evidence(EvidenceKind::GitState, &target) {
                None
            } else {
                Some(EvidenceGateViolation {
                    kind: EvidenceKind::GitState,
                    reason: "git_commit requires a fresh git state inspection first".to_string(),
                    coaching: format!(
                        "Inspect the repository state with `git_info` for {}, then retry `git_commit`.",
                        target.value
                    ),
                    target: Some(target.value),
                })
            }
        }
        _ => None,
    }
}

pub fn record_successful_tool_evidence(
    evidence_state: &mut EvidenceState,
    tool_name: &str,
    raw_arguments: &str,
    semantics: &ToolCallSemantics,
) {
    match tool_name {
        "read_file" => {
            if !semantics.target_hints.is_empty() {
                evidence_state.record_direct(
                    EvidenceKind::FileRead,
                    tool_name,
                    semantics.target_hints.clone(),
                );
            }
        }
        "git_info" => {
            evidence_state.record_direct(
                EvidenceKind::GitState,
                tool_name,
                git_scope_targets(raw_arguments, semantics),
            );
        }
        "service_status" => {
            let targets = semantics.target_hints.clone();
            if !targets.is_empty() {
                evidence_state.record_direct(EvidenceKind::ProcessState, tool_name, targets);
            }
        }
        _ => {}
    }
}

fn extract_string_arg(raw_arguments: &str, keys: &[&str]) -> Option<String> {
    let parsed = serde_json::from_str::<serde_json::Value>(raw_arguments).ok()?;
    let map = parsed.as_object()?;
    for key in keys {
        if let Some(value) = map.get(*key).and_then(|value| value.as_str()) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn path_target_hint(raw_arguments: &str) -> Option<ToolTargetHint> {
    extract_string_arg(raw_arguments, &["path", "file_path", "file", "filename"])
        .and_then(|path| ToolTargetHint::new(ToolTargetHintKind::Path, path))
}

fn git_scope_target_hint(raw_arguments: &str) -> ToolTargetHint {
    ToolTargetHint::new(
        ToolTargetHintKind::ProjectScope,
        extract_string_arg(raw_arguments, &["path", "repo_path", "repo_dir"])
            .unwrap_or_else(|| ".".to_string()),
    )
    .unwrap_or(ToolTargetHint {
        kind: ToolTargetHintKind::ProjectScope,
        value: ".".to_string(),
    })
}

fn git_scope_targets(raw_arguments: &str, semantics: &ToolCallSemantics) -> Vec<ToolTargetHint> {
    if semantics.target_hints.is_empty() {
        vec![git_scope_target_hint(raw_arguments)]
    } else {
        semantics.target_hints.clone()
    }
}

fn normalize_target_value(value: &str) -> String {
    let normalized = fs_utils::validate_path(value)
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_else(|_| value.to_string());
    normalized
        .trim()
        .trim_end_matches(['/', '\\'])
        .to_ascii_lowercase()
}

fn target_matches_expected(observed: &ToolTargetHint, expected: &ToolTargetHint) -> bool {
    let compatible_kind = matches!(
        (observed.kind, expected.kind),
        (ToolTargetHintKind::Path, ToolTargetHintKind::Path)
            | (
                ToolTargetHintKind::ProjectScope,
                ToolTargetHintKind::ProjectScope
            )
            | (ToolTargetHintKind::Path, ToolTargetHintKind::ProjectScope)
            | (ToolTargetHintKind::ProjectScope, ToolTargetHintKind::Path)
    );
    if !compatible_kind {
        return false;
    }

    let observed = normalize_target_value(&observed.value);
    let expected = normalize_target_value(&expected.value);
    !observed.is_empty()
        && !expected.is_empty()
        && (observed == expected
            || observed.starts_with(&format!("{expected}/"))
            || expected.starts_with(&format!("{observed}/")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{ToolCallSemantics, ToolTargetHintKind};

    #[test]
    fn edit_requires_prior_file_read() {
        let evidence = EvidenceState::default();
        let violation = assess_pre_execution_evidence_gate(
            "edit_file",
            r#"{"path":"/tmp/example.txt"}"#,
            &evidence,
        )
        .expect("edit_file should require prior read");
        assert_eq!(violation.kind, EvidenceKind::FileRead);
    }

    #[test]
    fn file_read_evidence_satisfies_edit_gate() {
        let mut evidence = EvidenceState::default();
        evidence.record_direct(
            EvidenceKind::FileRead,
            "read_file",
            vec![
                ToolTargetHint::new(ToolTargetHintKind::Path, "/tmp/example.txt")
                    .expect("path hint"),
            ],
        );
        let violation = assess_pre_execution_evidence_gate(
            "edit_file",
            r#"{"path":"/tmp/example.txt"}"#,
            &evidence,
        );
        assert!(violation.is_none());
    }

    #[test]
    fn git_commit_requires_git_state() {
        let evidence = EvidenceState::default();
        let violation =
            assess_pre_execution_evidence_gate("git_commit", r#"{"path":"."}"#, &evidence)
                .expect("git commit should require git state");
        assert_eq!(violation.kind, EvidenceKind::GitState);
    }

    #[test]
    fn git_info_registers_git_state() {
        let mut evidence = EvidenceState::default();
        record_successful_tool_evidence(
            &mut evidence,
            "git_info",
            r#"{"path":"."}"#,
            &ToolCallSemantics::observation(),
        );
        assert_eq!(evidence.records.len(), 1);
        assert_eq!(evidence.records[0].kind, EvidenceKind::GitState);
    }

    #[test]
    fn git_info_evidence_satisfies_git_commit_gate() {
        let mut evidence = EvidenceState::default();
        record_successful_tool_evidence(
            &mut evidence,
            "git_info",
            r#"{"path":"/tmp/example-repo"}"#,
            &ToolCallSemantics::observation(),
        );
        let violation = assess_pre_execution_evidence_gate(
            "git_commit",
            r#"{"path":"/tmp/example-repo","message":"checkpoint"}"#,
            &evidence,
        );
        assert!(violation.is_none());
    }
}
