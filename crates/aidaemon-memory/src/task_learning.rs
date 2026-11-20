//! Learning from finished tasks: a model reads what a task did and how it
//! ended, and the procedures and error solutions it finds are stored for
//! future tasks.
//!
//! It learns no facts about the owner. Its evidence is a model-written brief
//! and tool output, never the owner's words, so it is the lowest-provenance
//! writer there is; owner facts come from per-turn extraction.

use std::sync::Arc;

use aidaemon_core::{Message, ModelRequest, Provider};
use serde::Deserialize;
use tracing::{info, warn};

use crate::learning::{ErrorSolution, LearningStore, Procedure, procedure_keyed_name};

/// How long an extracted fact stays true, as declared by the extractor.
///
/// Background extractors see raw tool output, so without this they file run
/// state ("last command", "disk check result") as durable owner memory.
/// Transient observations are dropped at write time instead of being stored
/// and pinned. Anything unrecognised is durable, so an extractor that never
/// emits the field keeps its old behaviour.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FactLifetime {
    #[default]
    Durable,
    Transient,
}

impl FactLifetime {
    pub fn is_transient(self) -> bool {
        self == Self::Transient
    }
}

impl<'de> serde::Deserialize<'de> for FactLifetime {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = Option::<serde_json::Value>::deserialize(deserializer)?;
        Ok(match raw.as_ref().and_then(serde_json::Value::as_str) {
            Some(value) if value.trim().eq_ignore_ascii_case("transient") => Self::Transient,
            _ => Self::Durable,
        })
    }
}

/// Extractor prompt sentence that defines the `lifetime` field.
pub const FACT_LIFETIME_INSTRUCTION: &str = "Set \"lifetime\":\"transient\" on any fact that \
     describes state likely to change within days: the last command run, a single run's status \
     or result, current disk/CPU/queue numbers, a draft in progress, or today's plan; and on any \
     rule or constraint the user limits to this chat, this task or this request (\"for this chat\", \
     \"this time\", \"don't install anything for this one\"). Omit \"lifetime\" for facts that stay \
     true (names, preferences, project locations, tech stacks).";

/// What a finished task did, as the host renders it.
#[derive(Debug, Clone, Copy)]
pub struct FinishedTask<'a> {
    pub description: &'a str,
    pub result: Option<&'a str>,
    /// The task's activity, one line per tool call, result or status change.
    pub activity_log: &'a str,
    /// The project the task's work belongs to, with where its notes are
    /// kept: know-how that holds only in it is kept there (see
    /// [`crate::project`]). `None` learns exactly as before.
    pub project: Option<(
        &'a crate::project::ProjectNotes,
        &'a aidaemon_core::ProjectRef,
    )>,
}

/// What one task taught, and how much of it was stored.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TaskLearningReport {
    pub procedures: usize,
    pub error_solutions: usize,
    pub stored: usize,
    /// Of `stored`, how many went to the project's notes.
    pub project_notes: usize,
}

/// Extracts durable knowledge from finished tasks with a model.
#[derive(Clone)]
pub struct TaskLearner {
    provider: Arc<dyn Provider>,
    model: String,
}

impl TaskLearner {
    pub fn new(provider: Arc<dyn Provider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }

    /// Ask the model what `task` taught and store it in `store`. Extracted
    /// error solutions start with no credit: they earn it only when reused.
    pub async fn learn<S>(
        &self,
        store: &S,
        task: &FinishedTask<'_>,
    ) -> anyhow::Result<TaskLearningReport>
    where
        S: LearningStore + ?Sized,
    {
        let system_prompt = "You are a knowledge extraction system. Given a completed task and its activity log, \
        extract reusable know-how. Output ONLY a JSON object:\n\
        {\n  \"procedures\": [{\"name\": \"...\", \"trigger_pattern\": \"...\", \"steps\": [\"...\"]}],\n  \
        \"error_solutions\": [{\"error_pattern\": \"...\", \"solution_summary\": \"...\", \"domain\": \"...\"}]\n}\n\
        Only extract knowledge that would be useful for FUTURE tasks. If nothing worth remembering, return empty arrays.";
        let system_prompt = match task.project {
            Some((_, project)) => format!(
                "{system_prompt}\n\nThe task worked in the project \"{}\". Give every procedure and error \
                 solution a \"scope\": \"project\" when it holds only in this project (its commands, paths, \
                 configuration, conventions), \"general\" when it would help in other projects too.",
                project
                    .name
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            None => system_prompt.to_string(),
        };

        let user_prompt = format!(
            "## Task\n{}\n\n## Result\n{}\n\n## Activity Log\n{}",
            task.description,
            task.result.unwrap_or("(no result)"),
            task.activity_log
        );
        let request = ModelRequest {
            model: self.model.clone(),
            messages: vec![Message::system(system_prompt), Message::user(user_prompt)],
            ..Default::default()
        };
        let response = self.provider.complete(&request).await?;
        if response.text.trim().is_empty() {
            anyhow::bail!("Empty extraction response");
        }
        let extraction = parse_extraction_response(&response.text)?;

        let mut stored = 0usize;
        let mut project_notes = 0usize;
        // What holds only in the task's project becomes its notes.
        let in_project = |scope: &Option<String>| {
            task.project.filter(|_| {
                scope
                    .as_deref()
                    .is_some_and(|scope| scope.trim().eq_ignore_ascii_case("project"))
            })
        };
        for proc in &extraction.procedures {
            if proc.name.is_empty() || proc.steps.is_empty() {
                continue;
            }
            if let Some((notes, project)) = in_project(&proc.scope) {
                let mut body = String::new();
                if !proc.trigger_pattern.trim().is_empty() {
                    body.push_str(&format!("When: {}\n\n", proc.trigger_pattern.trim()));
                }
                for (i, step) in proc.steps.iter().enumerate() {
                    body.push_str(&format!("{}. {}\n", i + 1, step.trim()));
                }
                let note = crate::project::NoteWrite {
                    kind: crate::project::NoteKind::Procedure,
                    key: proc.name.clone(),
                    category: None,
                    description: Some(if proc.trigger_pattern.trim().is_empty() {
                        proc.name.clone()
                    } else {
                        proc.trigger_pattern.clone()
                    }),
                    body,
                    source: crate::project::NoteSource::Learned,
                };
                match notes.write(project, &note) {
                    Ok(_) => {
                        stored += 1;
                        project_notes += 1;
                    }
                    Err(e) => warn!(error = %e, "Failed to store project procedure note"),
                }
                continue;
            }
            let now = chrono::Utc::now();
            let procedure = Procedure {
                id: 0,
                name: procedure_keyed_name(&proc.name, &proc.steps),
                trigger_pattern: proc.trigger_pattern.clone(),
                steps: proc.steps.clone(),
                success_count: 1,
                failure_count: 0,
                avg_duration_secs: None,
                last_used_at: Some(now),
                created_at: now,
                updated_at: now,
            };
            match store.upsert_procedure(&procedure).await {
                Ok(_) => stored += 1,
                Err(e) => warn!(error = %e, "Failed to store extracted procedure"),
            }
        }

        for es in &extraction.error_solutions {
            if es.error_pattern.is_empty() || es.solution_summary.is_empty() {
                continue;
            }
            if let Some((notes, project)) = in_project(&es.scope) {
                let note = crate::project::NoteWrite {
                    kind: crate::project::NoteKind::Pitfall,
                    key: es.error_pattern.clone(),
                    category: None,
                    description: Some(es.error_pattern.clone()),
                    body: format!(
                        "Problem: {}\n\nWhat works: {}",
                        es.error_pattern.trim(),
                        es.solution_summary.trim()
                    ),
                    source: crate::project::NoteSource::Learned,
                };
                match notes.write(project, &note) {
                    Ok(_) => {
                        stored += 1;
                        project_notes += 1;
                    }
                    Err(e) => warn!(error = %e, "Failed to store project pitfall note"),
                }
                continue;
            }
            let now = chrono::Utc::now();
            let solution = ErrorSolution {
                id: 0,
                error_pattern: es.error_pattern.clone(),
                domain: (!es.domain.is_empty()).then(|| es.domain.clone()),
                solution_summary: es.solution_summary.clone(),
                solution_steps: None,
                success_count: 0,
                failure_count: 0,
                last_used_at: Some(now),
                created_at: now,
            };
            match store.insert_error_solution(&solution).await {
                Ok(_) => stored += 1,
                Err(e) => warn!(error = %e, "Failed to store extracted error solution"),
            }
        }

        let report = TaskLearningReport {
            procedures: extraction.procedures.len(),
            error_solutions: extraction.error_solutions.len(),
            stored,
            project_notes,
        };
        info!(?report, "Task knowledge extraction complete");
        Ok(report)
    }
}

/// Parsed extraction response from the LLM.
#[derive(Debug, Deserialize)]
struct ExtractionResponse {
    #[serde(default)]
    procedures: Vec<ExtractedProcedure>,
    #[serde(default)]
    error_solutions: Vec<ExtractedErrorSolution>,
}

#[derive(Debug, Deserialize)]
struct ExtractedProcedure {
    #[serde(default)]
    name: String,
    #[serde(default)]
    trigger_pattern: String,
    #[serde(default)]
    steps: Vec<String>,
    #[serde(default)]
    scope: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ExtractedErrorSolution {
    #[serde(default)]
    error_pattern: String,
    #[serde(default)]
    solution_summary: String,
    #[serde(default)]
    domain: String,
    #[serde(default)]
    scope: Option<String>,
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
        assert!(result.procedures.is_empty());
        assert!(result.error_solutions.is_empty());
    }

    #[test]
    fn test_parse_extraction_response_with_markdown_fences() {
        let response = "Here is the extraction:\n```json\n{\"facts\": [{\"category\": \"project\", \"key\": \"lang\", \"value\": \"Rust\"}], \"procedures\": [{\"name\": \"build\", \"steps\": [\"cargo build\"]}], \"error_solutions\": []}\n```";
        let result = parse_extraction_response(response).unwrap();
        assert_eq!(result.procedures[0].name, "build");
    }

    #[test]
    fn test_parse_extraction_response_partial_fields() {
        // Missing some fields — defaults should apply
        let response = r#"{"procedures": [{"name": "deploy"}]}"#;
        let result = parse_extraction_response(response).unwrap();
        assert!(result.procedures[0].steps.is_empty()); // default empty
        assert!(result.error_solutions.is_empty()); // missing → default empty
    }

    #[test]
    fn test_parse_extraction_response_invalid_json() {
        let response = "This is not JSON at all";
        assert!(parse_extraction_response(response).is_err());
    }
}

#[cfg(test)]
mod learner_tests {
    use std::sync::Mutex;

    use aidaemon_core::testing::{ScriptedProvider, text};
    use async_trait::async_trait;

    use super::*;
    /// Records what the learner stores.
    #[derive(Default)]
    struct Recorder {
        procedures: Mutex<Vec<Procedure>>,
        solutions: Mutex<Vec<ErrorSolution>>,
    }

    #[async_trait]
    impl LearningStore for Recorder {
        async fn upsert_procedure(&self, procedure: &Procedure) -> crate::Result<i64> {
            self.procedures.lock().unwrap().push(procedure.clone());
            Ok(1)
        }

        async fn insert_error_solution(&self, solution: &ErrorSolution) -> crate::Result<i64> {
            self.solutions.lock().unwrap().push(solution.clone());
            Ok(1)
        }
    }

    const REPLY: &str = r#"{"facts": [
        {"category": "project", "key": "stack", "value": "Rust"},
        {"category": "project", "key": "last_run", "value": "ok", "lifetime": "transient"}
      ],
      "procedures": [{"name": "deploy", "trigger_pattern": "deploy", "steps": ["build", "ship"]}],
      "error_solutions": [{"error_pattern": "connection refused", "solution_summary": "start the db"}]}"#;

    fn task() -> FinishedTask<'static> {
        FinishedTask {
            project: None,
            description: "Ship the release",
            result: Some("shipped"),
            activity_log: "TOOL_CALL: terminal args={}\n",
        }
    }

    #[tokio::test]
    async fn procedures_and_solutions_are_stored_and_facts_never_are() {
        let store = Recorder::default();
        let learner = TaskLearner::new(Arc::new(ScriptedProvider::new([text(REPLY)])), "mock");
        let report = learner.learn(&store, &task()).await.unwrap();

        assert_eq!((report.procedures, report.error_solutions), (1, 1));
        assert_eq!(report.stored, 2, "the reply's facts are ignored");
        let procedures = store.procedures.lock().unwrap();
        assert!(procedures[0].name.starts_with("deploy-"));
        let solutions = store.solutions.lock().unwrap();
        assert_eq!(solutions[0].success_count, 0, "extracted, not yet observed");
    }

    #[tokio::test]
    async fn project_only_know_how_becomes_the_projects_notes() {
        let reply = r#"{"procedures": [
            {"name": "deploy", "trigger_pattern": "deploy the site", "steps": ["make build", "make ship"], "scope": "project"},
            {"name": "bisect", "trigger_pattern": "find a regression", "steps": ["git bisect"], "scope": "general"}],
          "error_solutions": [
            {"error_pattern": "port 5433 in use", "solution_summary": "stop the old dev db", "scope": "project"},
            {"error_pattern": "connection refused", "solution_summary": "start the db"}]}"#;
        let dir = tempfile::tempdir().unwrap();
        let notes = crate::project::ProjectNotes::new(dir.path());
        let project = aidaemon_core::ProjectRef::new("project-a", "Acme Site", None).unwrap();
        let provider = Arc::new(ScriptedProvider::new([text(reply)]));
        let store = Recorder::default();
        let report = TaskLearner::new(provider.clone(), "mock")
            .learn(
                &store,
                &FinishedTask {
                    project: Some((&notes, &project)),
                    ..task()
                },
            )
            .await
            .unwrap();
        assert_eq!((report.stored, report.project_notes), (4, 2));
        assert_eq!(store.procedures.lock().unwrap().len(), 1);
        assert_eq!(store.solutions.lock().unwrap().len(), 1);
        let entries = notes.entries(&project);
        let kinds: Vec<_> = entries.iter().map(|e| (e.kind, e.name.clone())).collect();
        assert_eq!(
            kinds,
            [
                (
                    Some(crate::project::NoteKind::Procedure),
                    "deploy".to_string()
                ),
                (
                    Some(crate::project::NoteKind::Pitfall),
                    "port_5433_in_use".to_string()
                ),
            ]
        );
        let deploy = notes
            .read(&project, crate::project::NoteKind::Procedure, "deploy")
            .unwrap();
        assert!(deploy.contains("1. make build\n2. make ship"), "{deploy}");
        assert!(
            provider.requests()[0].messages[0]
                .content
                .contains("\"scope\"")
        );
        // Without a project, scopes are ignored and the prompt is unchanged.
        let provider = Arc::new(ScriptedProvider::new([text(reply)]));
        let store = Recorder::default();
        TaskLearner::new(provider.clone(), "mock")
            .learn(&store, &task())
            .await
            .unwrap();
        assert_eq!(store.procedures.lock().unwrap().len(), 2);
        assert!(!provider.requests()[0].messages[0].content.contains("scope"));
    }
}
