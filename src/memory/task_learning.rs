//! V3 Phase 4: Knowledge extraction from completed tasks.
//!
//! After an executor completes a task, this module reads activity logs and uses
//! an LLM to extract durable knowledge (facts, procedures, error solutions) that
//! can benefit future tasks.

use std::sync::Arc;

use serde::Deserialize;
use serde_json::json;
use tracing::{info, warn};

use crate::traits::{ModelProvider, StateStore, TaskV3};
use crate::types::FactPrivacy;

/// Maximum size of activity log text sent to the extraction prompt.
const MAX_ACTIVITY_LOG_CHARS: usize = 4096;

/// Extract durable knowledge from a completed task's activity log.
///
/// Reads the task's activity records, builds a prompt asking the LLM to extract
/// facts, procedures, and error solutions, then stores them in the state store.
pub async fn extract_task_knowledge(
    state: Arc<dyn StateStore>,
    provider: Arc<dyn ModelProvider>,
    model: String,
    task: TaskV3,
) -> anyhow::Result<()> {
    // Derive channel provenance from the parent goal (tasks do not carry session_id).
    let derived_channel_id: Option<String> = state
        .get_goal_v3(&task.goal_id)
        .await
        .ok()
        .flatten()
        .and_then(|g| crate::memory::derive_channel_id_from_session(&g.session_id));

    // Fetch activity log for this task
    let activities = state.get_task_activities_v3(&task.id).await?;
    if activities.is_empty() {
        info!(task_id = %task.id, "No activities to extract knowledge from");
        return Ok(());
    }

    // Build activity log text (truncated)
    let mut activity_log = String::new();
    for act in &activities {
        let line = match act.activity_type.as_str() {
            "tool_call" => {
                let name = act.tool_name.as_deref().unwrap_or("unknown");
                let args_preview = act
                    .tool_args
                    .as_deref()
                    .map(|a| &a[..a.len().min(200)])
                    .unwrap_or("");
                format!("TOOL_CALL: {} args={}\n", name, args_preview)
            }
            "tool_result" => {
                let name = act.tool_name.as_deref().unwrap_or("unknown");
                let success = act.success.unwrap_or(false);
                let result_preview = act
                    .result
                    .as_deref()
                    .map(|r| &r[..r.len().min(300)])
                    .unwrap_or("");
                format!(
                    "TOOL_RESULT: {} success={} result={}\n",
                    name, success, result_preview
                )
            }
            "status_change" => {
                let result_text = act.result.as_deref().unwrap_or("");
                format!("STATUS: {}\n", result_text)
            }
            _ => {
                format!(
                    "{}: {}\n",
                    act.activity_type,
                    act.result.as_deref().unwrap_or("")
                )
            }
        };

        if activity_log.len() + line.len() > MAX_ACTIVITY_LOG_CHARS {
            activity_log.push_str("... (truncated)\n");
            break;
        }
        activity_log.push_str(&line);
    }

    let result_text = task.result.as_deref().unwrap_or("(no result)");

    let system_prompt = "You are a knowledge extraction system. Given a completed task and its activity log, \
        extract durable knowledge worth remembering. Output ONLY a JSON object:\n\
        {\n  \"facts\": [{\"category\": \"...\", \"key\": \"...\", \"value\": \"...\"}],\n  \
        \"procedures\": [{\"name\": \"...\", \"trigger_pattern\": \"...\", \"steps\": [\"...\"]}],\n  \
        \"error_solutions\": [{\"error_pattern\": \"...\", \"solution_summary\": \"...\", \"domain\": \"...\"}]\n}\n\
        Categories for facts: project, technical, preference, behavior.\n\
        Only extract knowledge that would be useful for FUTURE tasks. If nothing worth remembering, return empty arrays.";

    let user_prompt = format!(
        "## Task\n{}\n\n## Result\n{}\n\n## Activity Log\n{}",
        task.description, result_text, activity_log
    );

    let messages = vec![
        json!({"role": "system", "content": system_prompt}),
        json!({"role": "user", "content": user_prompt}),
    ];

    let response = provider.chat(&model, &messages, &[]).await?;

    // Track token usage for task learning LLM calls
    if let Some(usage) = &response.usage {
        let _ = state
            .record_token_usage("background:task_learning", usage)
            .await;
    }

    let response_text = response
        .content
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("Empty extraction response"))?;

    // Parse response
    let extraction = parse_extraction_response(response_text)?;

    // Store extracted knowledge
    let mut stored = 0usize;

    for fact in &extraction.facts {
        if fact.category.is_empty() || fact.key.is_empty() || fact.value.is_empty() {
            continue;
        }
        if let Err(e) = state
            .upsert_fact(
                &fact.category,
                &fact.key,
                &fact.value,
                "v3_task_learning",
                derived_channel_id.as_deref(),
                FactPrivacy::Channel,
            )
            .await
        {
            warn!(error = %e, "Failed to store extracted fact");
        } else {
            stored += 1;
        }
    }

    for proc in &extraction.procedures {
        if proc.name.is_empty() || proc.steps.is_empty() {
            continue;
        }
        let procedure = crate::traits::Procedure {
            id: 0,
            name: proc.name.clone(),
            trigger_pattern: proc.trigger_pattern.clone(),
            steps: proc.steps.clone(),
            success_count: 1,
            failure_count: 0,
            avg_duration_secs: None,
            last_used_at: Some(chrono::Utc::now()),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        if let Err(e) = state.upsert_procedure(&procedure).await {
            warn!(error = %e, "Failed to store extracted procedure");
        } else {
            stored += 1;
        }
    }

    for es in &extraction.error_solutions {
        if es.error_pattern.is_empty() || es.solution_summary.is_empty() {
            continue;
        }
        let solution = crate::traits::ErrorSolution {
            id: 0,
            error_pattern: es.error_pattern.clone(),
            domain: if es.domain.is_empty() {
                None
            } else {
                Some(es.domain.clone())
            },
            solution_summary: es.solution_summary.clone(),
            solution_steps: None,
            success_count: 1,
            failure_count: 0,
            last_used_at: Some(chrono::Utc::now()),
            created_at: chrono::Utc::now(),
        };
        if let Err(e) = state.insert_error_solution(&solution).await {
            warn!(error = %e, "Failed to store extracted error solution");
        } else {
            stored += 1;
        }
    }

    info!(
        task_id = %task.id,
        facts = extraction.facts.len(),
        procedures = extraction.procedures.len(),
        error_solutions = extraction.error_solutions.len(),
        stored,
        "V3 task knowledge extraction complete"
    );

    Ok(())
}

/// Parsed extraction response from the LLM.
#[derive(Debug, Deserialize)]
struct ExtractionResponse {
    #[serde(default)]
    facts: Vec<ExtractedFact>,
    #[serde(default)]
    procedures: Vec<ExtractedProcedure>,
    #[serde(default)]
    error_solutions: Vec<ExtractedErrorSolution>,
}

#[derive(Debug, Deserialize)]
struct ExtractedFact {
    #[serde(default)]
    category: String,
    #[serde(default)]
    key: String,
    #[serde(default)]
    value: String,
}

#[derive(Debug, Deserialize)]
struct ExtractedProcedure {
    #[serde(default)]
    name: String,
    #[serde(default)]
    trigger_pattern: String,
    #[serde(default)]
    steps: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ExtractedErrorSolution {
    #[serde(default)]
    error_pattern: String,
    #[serde(default)]
    solution_summary: String,
    #[serde(default)]
    domain: String,
}

/// Parse the LLM extraction response, handling markdown fences and partial JSON.
fn parse_extraction_response(text: &str) -> anyhow::Result<ExtractionResponse> {
    // Try to extract JSON from markdown code fences first
    let json_str = if let Some(start) = text.find("```json") {
        let after = &text[start + 7..];
        if let Some(end) = after.find("```") {
            after[..end].trim()
        } else {
            text.trim()
        }
    } else if let Some(start) = text.find("```") {
        let after = &text[start + 3..];
        if let Some(end) = after.find("```") {
            after[..end].trim()
        } else {
            text.trim()
        }
    } else {
        text.trim()
    };

    // Try to find a JSON object in the text
    let json_str = if let Some(start) = json_str.find('{') {
        if let Some(end) = json_str.rfind('}') {
            &json_str[start..=end]
        } else {
            json_str
        }
    } else {
        json_str
    };

    serde_json::from_str(json_str).map_err(|e| {
        anyhow::anyhow!(
            "Failed to parse extraction response: {} (text: {})",
            e,
            &text[..text.len().min(200)]
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_extraction_response_full() {
        let response = r#"{"facts": [{"category": "technical", "key": "database", "value": "Uses PostgreSQL 15"}], "procedures": [{"name": "deploy", "trigger_pattern": "deploy to prod", "steps": ["build", "test", "deploy"]}], "error_solutions": [{"error_pattern": "connection refused", "solution_summary": "Check if DB is running", "domain": "database"}]}"#;

        let result = parse_extraction_response(response).unwrap();
        assert_eq!(result.facts.len(), 1);
        assert_eq!(result.facts[0].category, "technical");
        assert_eq!(result.facts[0].key, "database");
        assert_eq!(result.procedures.len(), 1);
        assert_eq!(result.procedures[0].name, "deploy");
        assert_eq!(result.procedures[0].steps.len(), 3);
        assert_eq!(result.error_solutions.len(), 1);
        assert_eq!(
            result.error_solutions[0].error_pattern,
            "connection refused"
        );
    }

    #[test]
    fn test_parse_extraction_response_empty_arrays() {
        let response = r#"{"facts": [], "procedures": [], "error_solutions": []}"#;
        let result = parse_extraction_response(response).unwrap();
        assert!(result.facts.is_empty());
        assert!(result.procedures.is_empty());
        assert!(result.error_solutions.is_empty());
    }

    #[test]
    fn test_parse_extraction_response_with_markdown_fences() {
        let response = "Here is the extraction:\n```json\n{\"facts\": [{\"category\": \"project\", \"key\": \"lang\", \"value\": \"Rust\"}], \"procedures\": [], \"error_solutions\": []}\n```";
        let result = parse_extraction_response(response).unwrap();
        assert_eq!(result.facts.len(), 1);
        assert_eq!(result.facts[0].value, "Rust");
    }

    #[test]
    fn test_parse_extraction_response_partial_fields() {
        // Missing some fields — defaults should apply
        let response = r#"{"facts": [{"category": "technical"}], "procedures": []}"#;
        let result = parse_extraction_response(response).unwrap();
        assert_eq!(result.facts.len(), 1);
        assert_eq!(result.facts[0].key, ""); // default empty
        assert!(result.error_solutions.is_empty()); // missing → default empty
    }

    #[test]
    fn test_parse_extraction_response_invalid_json() {
        let response = "This is not JSON at all";
        assert!(parse_extraction_response(response).is_err());
    }
}
