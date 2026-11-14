//! Learning from finished tasks: a model reads what a task did and how it
//! ended, and the durable knowledge it finds (facts, procedures, error
//! solutions) is stored for future tasks.

use std::sync::Arc;

use aidaemon_core::{Message, ModelRequest, Provider};
use serde::Deserialize;
use tracing::{debug, info, warn};

use crate::learning::{ErrorSolution, LearningStore, Procedure, procedure_keyed_name};
use crate::records::FactPrivacy;
use crate::stores::FactStore;

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
     or result, current disk/CPU/queue numbers, a draft in progress, or today's plan. Omit \
     \"lifetime\" for facts that stay true (names, preferences, project locations, tech stacks).";

/// What a finished task did, as the host renders it.
#[derive(Debug, Clone, Copy)]
pub struct FinishedTask<'a> {
    pub description: &'a str,
    pub result: Option<&'a str>,
    /// The task's activity, one line per tool call, result or status change.
    pub activity_log: &'a str,
    /// The channel the knowledge is scoped to.
    pub channel_id: Option<&'a str>,
    /// Whether facts are learned. A recurring task re-runs the same brief,
    /// so its "facts" restate its own instructions; its procedures and
    /// error solutions still carry over.
    pub learn_facts: bool,
}

/// What one task taught, and how much of it was stored.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TaskLearningReport {
    pub facts: usize,
    pub procedures: usize,
    pub error_solutions: usize,
    pub stored: usize,
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

    /// Ask the model what `task` taught and store it in `store`. Facts are
    /// channel-private and come from `task_learning`; transient ones are
    /// dropped. Extracted error solutions start with no credit: they earn
    /// it only when reused.
    pub async fn learn<S>(
        &self,
        store: &S,
        task: &FinishedTask<'_>,
    ) -> anyhow::Result<TaskLearningReport>
    where
        S: FactStore + LearningStore + ?Sized,
    {
        let system_prompt = "You are a knowledge extraction system. Given a completed task and its activity log, \
        extract durable knowledge worth remembering. Output ONLY a JSON object:\n\
        {\n  \"facts\": [{\"category\": \"...\", \"key\": \"...\", \"value\": \"...\", \"graph\": {\"entities\": [], \"relationships\": []}}],\n  \
        \"procedures\": [{\"name\": \"...\", \"trigger_pattern\": \"...\", \"steps\": [\"...\"]}],\n  \
        \"error_solutions\": [{\"error_pattern\": \"...\", \"solution_summary\": \"...\", \"domain\": \"...\"}]\n}\n\
        Categories for facts: project, technical, preference, behavior. Graph entities use local_id, name, entity_type, aliases, confidence; relationships use source_id, target_id, relation, confidence. Only include graph data directly supported by the task evidence. Never infer entity types from category or key names.\n\
        Only extract knowledge that would be useful for FUTURE tasks. If nothing worth remembering, return empty arrays.";
        let system_prompt = format!(
            "{system_prompt}\nFacts may carry a \"lifetime\" field. {}",
            FACT_LIFETIME_INSTRUCTION
        );

        let user_prompt = format!(
            "## Task\n{}\n\n## Result\n{}\n\n## Activity Log\n{}",
            task.description,
            task.result.unwrap_or("(no result)"),
            task.activity_log
        );
        let request = ModelRequest {
            model: self.model.clone(),
            messages: vec![
                Message::system(system_prompt),
                Message::user(user_prompt.clone()),
            ],
            ..Default::default()
        };
        let response = self.provider.complete(&request).await?;
        if response.text.trim().is_empty() {
            anyhow::bail!("Empty extraction response");
        }
        let extraction = parse_extraction_response(&response.text)?;

        let mut stored = 0usize;
        for fact in extraction.facts.iter().filter(|_| task.learn_facts) {
            if fact.category.is_empty() || fact.key.is_empty() || fact.value.is_empty() {
                continue;
            }
            if fact.lifetime.is_transient() {
                continue;
            }
            if let Err(e) = store
                .upsert_fact(
                    &fact.category,
                    &fact.key,
                    &fact.value,
                    "task_learning",
                    task.channel_id,
                    FactPrivacy::Channel,
                )
                .await
            {
                warn!(error = %e, "Failed to store extracted fact");
                continue;
            }
            let excerpt = crate::text::truncate_chars(&user_prompt, 1000);
            if let Err(e) = store
                .project_extracted_fact_graph(&fact.category, &fact.key, &excerpt, &fact.graph)
                .await
            {
                debug!(error = %e, key = fact.key, "Failed to project task-learning graph");
            }
            stored += 1;
        }

        for proc in &extraction.procedures {
            if proc.name.is_empty() || proc.steps.is_empty() {
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
            facts: extraction.facts.len(),
            procedures: extraction.procedures.len(),
            error_solutions: extraction.error_solutions.len(),
            stored,
        };
        info!(?report, "Task knowledge extraction complete");
        Ok(report)
    }
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
    #[serde(default)]
    graph: crate::records::ExtractedMemoryGraph,
    #[serde(default)]
    lifetime: FactLifetime,
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

#[cfg(test)]
mod learner_tests {
    use std::sync::Mutex;

    use aidaemon_core::testing::{ScriptedProvider, text};
    use async_trait::async_trait;

    use super::*;
    use crate::records::Fact;

    /// A stored fact: key, value, channel and privacy.
    type StoredFact = (String, String, Option<String>, FactPrivacy);

    /// Records what the learner stores.
    #[derive(Default)]
    struct Recorder {
        facts: Mutex<Vec<StoredFact>>,
        procedures: Mutex<Vec<Procedure>>,
        solutions: Mutex<Vec<ErrorSolution>>,
    }

    #[async_trait]
    impl FactStore for Recorder {
        #[allow(clippy::too_many_arguments)]
        async fn upsert_fact_with_provenance(
            &self,
            _category: &str,
            key: &str,
            value: &str,
            source: &str,
            channel_id: Option<&str>,
            privacy: FactPrivacy,
            _first_seen_at: Option<chrono::DateTime<chrono::Utc>>,
            _source_excerpt: Option<&str>,
        ) -> anyhow::Result<()> {
            assert_eq!(source, "task_learning");
            self.facts.lock().unwrap().push((
                key.to_string(),
                value.to_string(),
                channel_id.map(str::to_string),
                privacy,
            ));
            Ok(())
        }

        async fn get_facts(&self, _category: Option<&str>) -> anyhow::Result<Vec<Fact>> {
            Ok(Vec::new())
        }
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

    fn task(learn_facts: bool) -> FinishedTask<'static> {
        FinishedTask {
            description: "Ship the release",
            result: Some("shipped"),
            activity_log: "TOOL_CALL: terminal args={}\n",
            channel_id: Some("telegram:synthetic-chat"),
            learn_facts,
        }
    }

    #[tokio::test]
    async fn durable_knowledge_is_stored_and_transient_facts_are_dropped() {
        let store = Recorder::default();
        let learner = TaskLearner::new(Arc::new(ScriptedProvider::new([text(REPLY)])), "mock");
        let report = learner.learn(&store, &task(true)).await.unwrap();

        assert_eq!(
            (report.facts, report.procedures, report.error_solutions),
            (2, 1, 1)
        );
        assert_eq!(report.stored, 3);
        let facts = store.facts.lock().unwrap();
        assert_eq!(facts.len(), 1, "the transient fact is dropped");
        assert_eq!(facts[0].0, "stack");
        assert_eq!(facts[0].2.as_deref(), Some("telegram:synthetic-chat"));
        assert_eq!(facts[0].3, FactPrivacy::Channel);
        let procedures = store.procedures.lock().unwrap();
        assert!(procedures[0].name.starts_with("deploy-"));
        let solutions = store.solutions.lock().unwrap();
        assert_eq!(solutions[0].success_count, 0, "extracted, not yet observed");
    }

    #[tokio::test]
    async fn a_recurring_task_learns_no_facts() {
        let store = Recorder::default();
        let learner = TaskLearner::new(Arc::new(ScriptedProvider::new([text(REPLY)])), "mock");
        let report = learner.learn(&store, &task(false)).await.unwrap();
        assert_eq!(report.stored, 2);
        assert!(store.facts.lock().unwrap().is_empty());
    }
}
