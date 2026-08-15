//! Typed inquiry/evidence routing.
//!
//! This module deliberately matches protocol metadata rather than request
//! wording. The task assessor names material information needs; tools advertise
//! which kinds of evidence their exact operation can produce; completion joins
//! the two through durable receipts.

use std::collections::HashSet;

use crate::traits::{
    EvidenceAuthority, EvidencePurpose, EvidenceTemporalScope, RequestEvidenceRequirement,
    ToolEvidenceCapability, ToolSemanticScope, ToolTargetHint, ToolTargetHintKind,
};

fn capability(
    scope: ToolSemanticScope,
    purposes: &[EvidencePurpose],
    authority: EvidenceAuthority,
    temporal_scope: EvidenceTemporalScope,
) -> ToolEvidenceCapability {
    ToolEvidenceCapability::new(scope, purposes, authority, temporal_scope)
}

fn action_from_arguments(arguments: &str) -> Option<String> {
    string_argument(arguments, "action")
}

fn string_argument(arguments: &str, key: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()?
        .get(key)?
        .as_str()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

/// Evidence affordances for a successful result from one exact tool call.
/// Dispatch still decides whether the call was an observation; this metadata
/// only states what a substantive result from that observation can support.
pub(in crate::agent) fn evidence_capabilities_for_tool_call(
    tool_name: &str,
    arguments: &str,
) -> Vec<ToolEvidenceCapability> {
    use EvidenceAuthority::{Advisory, Canonical, Direct};
    use EvidencePurpose::{
        Attribution, CausalExplanation, Content, CurrentState, HistoricalRecord, Outcome,
    };
    use EvidenceTemporalScope::{Both, Current, Historical};
    use ToolSemanticScope::{
        ConversationHistory, ExternalRemote, GoalState, HostLocal, LocalWorkspace, UserMemory,
    };

    match tool_name {
        "goal_trace" | "tool_trace" => {
            let targeted = string_argument(arguments, "goal_id").is_some()
                || string_argument(arguments, "task_id").is_some();
            if targeted {
                vec![capability(
                    GoalState,
                    &[
                        CurrentState,
                        HistoricalRecord,
                        Outcome,
                        Attribution,
                        CausalExplanation,
                        Content,
                    ],
                    Canonical,
                    Both,
                )]
            } else {
                // An unfiltered recent-goal listing is useful discovery, but
                // it cannot prove attribution or causality for a particular
                // prior execution.
                vec![capability(
                    GoalState,
                    &[CurrentState, Content],
                    Canonical,
                    Current,
                )]
            }
        }
        "manage_mandates" => {
            let action = action_from_arguments(arguments);
            let targeted = string_argument(arguments, "mandate_id").is_some();
            match action.as_deref() {
                Some("get")
                    if targeted
                        && string_argument(arguments, "section").as_deref() == Some("history") =>
                {
                    vec![capability(
                        GoalState,
                        &[
                            HistoricalRecord,
                            Outcome,
                            Attribution,
                            CausalExplanation,
                            Content,
                        ],
                        Canonical,
                        Historical,
                    )]
                }
                Some("list_intentions") if targeted => vec![capability(
                    GoalState,
                    &[HistoricalRecord, Attribution, CausalExplanation, Content],
                    Canonical,
                    Historical,
                )],
                Some("get") if targeted => vec![capability(
                    GoalState,
                    &[CurrentState, Outcome, Content],
                    Canonical,
                    Current,
                )],
                Some("list") | None => vec![capability(
                    GoalState,
                    &[CurrentState, Content],
                    Canonical,
                    Current,
                )],
                _ => Vec::new(),
            }
        }
        "manage_goal_tasks" | "scheduled_goal_runs" | "scheduled_goals" => vec![capability(
            GoalState,
            &[
                CurrentState,
                HistoricalRecord,
                Outcome,
                Attribution,
                Content,
            ],
            Canonical,
            Both,
        )],
        "search_history" | "read_channel_history" => vec![capability(
            ConversationHistory,
            &[HistoricalRecord, Content, Attribution],
            Canonical,
            Historical,
        )],
        "manage_memories" => match action_from_arguments(arguments).as_deref() {
            Some("list_goals" | "list_scheduled" | "list_scheduled_matching") => vec![capability(
                GoalState,
                &[CurrentState, Content],
                Canonical,
                Current,
            )],
            Some("diagnose_scheduled") => vec![capability(
                GoalState,
                &[
                    CurrentState,
                    HistoricalRecord,
                    Outcome,
                    CausalExplanation,
                    Content,
                ],
                Canonical,
                Both,
            )],
            Some("search_episodes") => vec![capability(
                UserMemory,
                &[HistoricalRecord, Content, Attribution],
                Advisory,
                Historical,
            )],
            Some("list" | "search") | None => vec![capability(
                UserMemory,
                &[CurrentState, Content, Attribution],
                Advisory,
                Current,
            )],
            _ => Vec::new(),
        },
        "manage_people" | "remember_fact" | "share_memory" => vec![capability(
            UserMemory,
            &[CurrentState, HistoricalRecord, Content, Attribution],
            Advisory,
            Both,
        )],
        "http_request" | "web_fetch" | "browser" => vec![capability(
            ExternalRemote,
            &[CurrentState, Content, Outcome],
            Direct,
            Current,
        )],
        "read_node_health" | "read_node_sensors" => vec![capability(
            ExternalRemote,
            &[CurrentState, Content],
            Direct,
            Current,
        )],
        "send_node_audio" => vec![capability(ExternalRemote, &[Outcome], Direct, Current)],
        "web_search" => vec![capability(
            ExternalRemote,
            &[CurrentState, Content],
            Advisory,
            Current,
        )],
        "read_file" | "search_files" | "project_inspect" => vec![capability(
            LocalWorkspace,
            &[CurrentState, Content, CausalExplanation],
            Direct,
            Current,
        )],
        "git_info" => vec![capability(
            LocalWorkspace,
            &[
                CurrentState,
                HistoricalRecord,
                Outcome,
                Attribution,
                Content,
            ],
            Canonical,
            Both,
        )],
        "terminal" | "run_command" => vec![
            capability(
                LocalWorkspace,
                &[CurrentState, Content, Outcome, CausalExplanation],
                Direct,
                Current,
            ),
            capability(
                HostLocal,
                &[CurrentState, Content, Outcome, CausalExplanation],
                Direct,
                Current,
            ),
        ],
        "system_info" | "service_status" | "check_environment" => vec![capability(
            HostLocal,
            &[CurrentState, Content, Outcome, CausalExplanation],
            Direct,
            Current,
        )],
        "self_diagnose" | "diagnose" => vec![capability(
            HostLocal,
            &[
                CurrentState,
                HistoricalRecord,
                Outcome,
                Attribution,
                CausalExplanation,
            ],
            Canonical,
            Both,
        )],
        _ => Vec::new(),
    }
}

/// Derive evidence scope from exact resource identities advertised by a tool.
/// This lets dynamic/custom tools participate without teaching the controller
/// their names. Resource IDs remain unclassified because their domain is
/// intentionally opaque outside the session registry.
pub(in crate::agent) fn evidence_capabilities_from_target_hints(
    target_hints: &[ToolTargetHint],
) -> Vec<ToolEvidenceCapability> {
    let mut capabilities = Vec::new();
    for hint in target_hints {
        let capability = match hint.kind {
            ToolTargetHintKind::Url => capability(
                ToolSemanticScope::ExternalRemote,
                &[
                    EvidencePurpose::CurrentState,
                    EvidencePurpose::Content,
                    EvidencePurpose::Outcome,
                ],
                EvidenceAuthority::Direct,
                EvidenceTemporalScope::Current,
            ),
            ToolTargetHintKind::Path | ToolTargetHintKind::ProjectScope => capability(
                ToolSemanticScope::LocalWorkspace,
                &[
                    EvidencePurpose::CurrentState,
                    EvidencePurpose::Content,
                    EvidencePurpose::CausalExplanation,
                ],
                EvidenceAuthority::Direct,
                EvidenceTemporalScope::Current,
            ),
            ToolTargetHintKind::ResourceId => continue,
        };
        if !capabilities.contains(&capability) {
            capabilities.push(capability);
        }
    }
    capabilities
}

/// Union of action-specific affordances used only to recommend candidate
/// tools. Exact receipt credit always uses `evidence_capabilities_for_tool_call`.
fn static_evidence_capabilities(tool_name: &str) -> Vec<ToolEvidenceCapability> {
    let representative_calls: &[&str] = match tool_name {
        "goal_trace" | "tool_trace" => &[r#"{"action":"tool_trace","task_id":"candidate"}"#],
        "manage_mandates" => &[
            r#"{"action":"list"}"#,
            r#"{"action":"get","mandate_id":"candidate","section":"history"}"#,
            r#"{"action":"list_intentions","mandate_id":"candidate"}"#,
        ],
        "manage_memories" => &[
            r#"{"action":"search"}"#,
            r#"{"action":"search_episodes"}"#,
            r#"{"action":"list_goals"}"#,
            r#"{"action":"diagnose_scheduled"}"#,
        ],
        _ => &["{}"],
    };
    let mut capabilities = Vec::new();
    for arguments in representative_calls {
        for capability in evidence_capabilities_for_tool_call(tool_name, arguments) {
            if !capabilities.contains(&capability) {
                capabilities.push(capability);
            }
        }
    }
    capabilities
}

/// Whether the controller has enough static evidence metadata to conclude
/// that a visible tool cannot satisfy an inquiry. Custom/dynamic tools may
/// advertise argument-dependent semantics only when they execute, so an
/// unknown name must remain a possible candidate rather than being declared
/// unavailable prematurely.
pub(in crate::agent) fn has_static_evidence_model(tool_name: &str) -> bool {
    !static_evidence_capabilities(tool_name).is_empty()
}

pub(in crate::agent) fn capability_supports_requirement(
    capability: &ToolEvidenceCapability,
    requirement: &RequestEvidenceRequirement,
) -> bool {
    requirement.acceptable_scopes.contains(&capability.scope)
        && capability.purposes.contains(&requirement.purpose)
        && capability
            .authority
            .satisfies(requirement.minimum_authority)
        && capability
            .temporal_scope
            .satisfies(requirement.temporal_scope)
}

pub(in crate::agent) fn tool_call_supports_any_requirement(
    tool_name: &str,
    arguments: &str,
    requirements: &[RequestEvidenceRequirement],
) -> bool {
    let capabilities = evidence_capabilities_for_tool_call(tool_name, arguments);
    requirements.iter().any(|requirement| {
        capabilities
            .iter()
            .any(|capability| capability_supports_requirement(capability, requirement))
    })
}

pub(in crate::agent) fn candidate_tools_for_requirements<'a>(
    requirements: &[RequestEvidenceRequirement],
    visible_tool_names: impl IntoIterator<Item = &'a str>,
) -> Vec<String> {
    let mut scored = visible_tool_names
        .into_iter()
        .filter_map(|name| {
            let capabilities = static_evidence_capabilities(name);
            let coverage = requirements
                .iter()
                .filter(|requirement| {
                    capabilities
                        .iter()
                        .any(|capability| capability_supports_requirement(capability, requirement))
                })
                .count();
            (coverage > 0).then(|| (coverage, name.to_string()))
        })
        .collect::<Vec<_>>();
    scored.sort_by(|(left_coverage, left_name), (right_coverage, right_name)| {
        right_coverage
            .cmp(left_coverage)
            .then_with(|| left_name.cmp(right_name))
    });
    scored.dedup_by(|left, right| left.1 == right.1);
    scored.into_iter().take(10).map(|(_, name)| name).collect()
}

/// Typed epistemic uncertainty derived after semantic assessment. Operational
/// risk remains separate; this score reflects how many independent evidence
/// obligations and source classes must be reconciled.
pub(in crate::agent) fn epistemic_uncertainty(requirements: &[RequestEvidenceRequirement]) -> f32 {
    if requirements.is_empty() {
        return 0.0;
    }

    let mut score = 0.18_f32 + 0.11 * requirements.len().saturating_sub(1) as f32;
    let scopes = requirements
        .iter()
        .flat_map(|requirement| requirement.acceptable_scopes.iter().copied())
        .collect::<HashSet<_>>();
    if scopes.len() > 1 {
        score += 0.12;
    }
    if requirements.iter().any(|requirement| {
        matches!(
            requirement.purpose,
            EvidencePurpose::Attribution | EvidencePurpose::CausalExplanation
        )
    }) {
        score += 0.14;
    }
    if requirements
        .iter()
        .any(|requirement| requirement.minimum_authority == EvidenceAuthority::Canonical)
    {
        score += 0.10;
    }
    score.clamp(0.0, 0.9)
}

pub(in crate::agent) fn describe_requirement(requirement: &RequestEvidenceRequirement) -> String {
    let scopes = requirement
        .acceptable_scopes
        .iter()
        .map(|scope| scope.as_str())
        .collect::<Vec<_>>()
        .join("|");
    let target = requirement
        .target
        .as_ref()
        .map(|target| format!("; target={}", target.value))
        .unwrap_or_default();
    format!(
        "{} [scope={}; purpose={}; authority={}; time={}{}]",
        requirement.summary,
        scopes,
        requirement.purpose.as_str(),
        requirement.minimum_authority.as_str(),
        requirement.temporal_scope.as_str(),
        target,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requirement(
        scope: ToolSemanticScope,
        purpose: EvidencePurpose,
        authority: EvidenceAuthority,
        temporal_scope: EvidenceTemporalScope,
    ) -> RequestEvidenceRequirement {
        RequestEvidenceRequirement {
            summary: "synthetic evidence need".to_string(),
            acceptable_scopes: vec![scope],
            purpose,
            minimum_authority: authority,
            temporal_scope,
            required_content_markers: Vec::new(),
            target: None,
        }
    }

    #[test]
    fn current_external_state_cannot_prove_canonical_execution_history() {
        let need = requirement(
            ToolSemanticScope::GoalState,
            EvidencePurpose::HistoricalRecord,
            EvidenceAuthority::Canonical,
            EvidenceTemporalScope::Historical,
        );
        let external = evidence_capabilities_for_tool_call("http_request", r#"{"method":"GET"}"#);
        assert!(!external
            .iter()
            .any(|capability| capability_supports_requirement(capability, &need)));
        let trace = evidence_capabilities_for_tool_call(
            "goal_trace",
            r#"{"action":"tool_trace","task_id":"synthetic-task"}"#,
        );
        assert!(trace
            .iter()
            .any(|capability| capability_supports_requirement(capability, &need)));
    }

    #[test]
    fn advisory_memory_cannot_satisfy_canonical_attribution() {
        let need = requirement(
            ToolSemanticScope::UserMemory,
            EvidencePurpose::Attribution,
            EvidenceAuthority::Canonical,
            EvidenceTemporalScope::Historical,
        );
        let memory = evidence_capabilities_for_tool_call(
            "manage_memories",
            r#"{"action":"search_episodes"}"#,
        );
        assert!(!memory
            .iter()
            .any(|capability| capability_supports_requirement(capability, &need)));
    }

    #[test]
    fn unfiltered_trace_listing_cannot_prove_specific_attribution() {
        let need = requirement(
            ToolSemanticScope::GoalState,
            EvidencePurpose::Attribution,
            EvidenceAuthority::Canonical,
            EvidenceTemporalScope::Historical,
        );
        let recent =
            evidence_capabilities_for_tool_call("goal_trace", r#"{"action":"tool_trace"}"#);
        assert!(!recent
            .iter()
            .any(|capability| capability_supports_requirement(capability, &need)));
        let targeted = evidence_capabilities_for_tool_call(
            "goal_trace",
            r#"{"action":"tool_trace","task_id":"synthetic-task"}"#,
        );
        assert!(targeted
            .iter()
            .any(|capability| capability_supports_requirement(capability, &need)));
    }

    #[test]
    fn exact_url_hint_gives_an_unknown_tool_direct_current_scope_only() {
        let hints =
            vec![
                ToolTargetHint::new(ToolTargetHintKind::Url, "https://example.test/state").unwrap(),
            ];
        let capabilities = evidence_capabilities_from_target_hints(&hints);
        assert!(capabilities.iter().any(|capability| {
            capability.scope == ToolSemanticScope::ExternalRemote
                && capability.authority == EvidenceAuthority::Direct
                && capability.purposes.contains(&EvidencePurpose::CurrentState)
        }));
        assert!(!capabilities.iter().any(|capability| {
            capability.purposes.contains(&EvidencePurpose::Attribution)
                || capability
                    .purposes
                    .contains(&EvidencePurpose::HistoricalRecord)
        }));
    }

    #[test]
    fn secondary_domain_tool_is_relevant_to_a_compound_evidence_contract() {
        let need = requirement(
            ToolSemanticScope::ExternalRemote,
            EvidencePurpose::CurrentState,
            EvidenceAuthority::Direct,
            EvidenceTemporalScope::Current,
        );
        assert!(tool_call_supports_any_requirement(
            "http_request",
            r#"{"method":"GET","url":"https://example.test/feed"}"#,
            &[need],
        ));
    }

    #[test]
    fn node_receipt_tools_cover_current_state_and_playback_outcome_needs() {
        let current_state = requirement(
            ToolSemanticScope::ExternalRemote,
            EvidencePurpose::CurrentState,
            EvidenceAuthority::Direct,
            EvidenceTemporalScope::Current,
        );
        let playback_outcome = requirement(
            ToolSemanticScope::ExternalRemote,
            EvidencePurpose::Outcome,
            EvidenceAuthority::Direct,
            EvidenceTemporalScope::Current,
        );

        assert!(tool_call_supports_any_requirement(
            "read_node_health",
            r#"{"node":"Synthetic Companion"}"#,
            std::slice::from_ref(&current_state),
        ));
        assert!(tool_call_supports_any_requirement(
            "send_node_audio",
            r#"{"node":"Synthetic Companion","text":"Hello"}"#,
            std::slice::from_ref(&playback_outcome),
        ));
        assert!(!tool_call_supports_any_requirement(
            "read_node_health",
            r#"{"node":"Synthetic Companion"}"#,
            &[playback_outcome],
        ));
    }

    #[test]
    fn candidate_ranking_prefers_tools_covering_more_needs() {
        let needs = vec![
            requirement(
                ToolSemanticScope::GoalState,
                EvidencePurpose::HistoricalRecord,
                EvidenceAuthority::Canonical,
                EvidenceTemporalScope::Historical,
            ),
            requirement(
                ToolSemanticScope::GoalState,
                EvidencePurpose::Attribution,
                EvidenceAuthority::Canonical,
                EvidenceTemporalScope::Historical,
            ),
        ];
        let candidates = candidate_tools_for_requirements(
            &needs,
            ["http_request", "manage_memories", "goal_trace"],
        );
        assert_eq!(candidates.first().map(String::as_str), Some("goal_trace"));
        assert!(candidates.iter().any(|name| name == "manage_memories"));
        assert!(!candidates.iter().any(|name| name == "http_request"));
    }

    #[test]
    fn multi_domain_attribution_has_material_uncertainty() {
        let needs = vec![
            requirement(
                ToolSemanticScope::ExternalRemote,
                EvidencePurpose::CurrentState,
                EvidenceAuthority::Direct,
                EvidenceTemporalScope::Current,
            ),
            requirement(
                ToolSemanticScope::GoalState,
                EvidencePurpose::Attribution,
                EvidenceAuthority::Canonical,
                EvidenceTemporalScope::Historical,
            ),
        ];
        assert!(epistemic_uncertainty(&needs) >= 0.6);
    }

    #[test]
    fn unknown_dynamic_tool_prevents_a_false_unavailable_conclusion() {
        assert!(has_static_evidence_model("web_search"));
        assert!(has_static_evidence_model("system_info"));
        assert!(!has_static_evidence_model("check_remote"));
    }
}
