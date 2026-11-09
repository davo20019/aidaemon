//! The one door for side calls: model calls the daemon makes outside the
//! agent loop. The gateway turns a [`SideCallSpec`] into a model request,
//! bounds every attempt and failover step by the spec's one total deadline,
//! and writes one `llm_call` row per attempt, failed attempts included.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_core as sdk;
use async_trait::async_trait;
use tokio::time::Instant;

use super::side_calls::{ModelChoice, SideCall, SideCallClass, SideCallSpec};
use super::{ModelBackend, SharedLlmRuntime};
use crate::events::EventStore;
use crate::providers::sdk_backed::{
    to_chat_messages, to_chat_options, to_provider_response, to_sdk_messages, to_sdk_response,
};
use crate::traits::StateStore;

/// Runs side calls: picks the model chain from the spec, applies its total
/// deadline and records every attempt.
#[derive(Clone)]
pub struct SideCallGateway {
    runtime: SharedLlmRuntime,
    state: Arc<dyn StateStore>,
    event_store: Arc<EventStore>,
}

/// A side call's answer.
#[derive(Debug, Clone)]
pub struct SideCallAnswer {
    #[allow(dead_code)] // used from Task 3
    pub text: String,
    /// The model that answered.
    #[allow(dead_code)] // used from Task 3
    pub model: String,
    /// Time from the request to the answer, across every attempt.
    #[allow(dead_code)] // used from Task 3
    pub elapsed: Duration,
}

/// Why a side call has no answer.
#[derive(Debug)]
pub enum SideCallError {
    Deadline(Duration),
    Provider(String),
    Empty,
}

impl std::fmt::Display for SideCallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Deadline(deadline) => write!(f, "side call exceeded its {deadline:?} deadline"),
            Self::Provider(error) => write!(f, "every model failed: {error}"),
            Self::Empty => write!(f, "the model returned no text"),
        }
    }
}

impl std::error::Error for SideCallError {}

/// One attempt on one model, as the gateway saw it.
struct Attempt {
    model: String,
    started: Instant,
    /// `None` while it runs, and for an attempt dropped before it ended
    /// (its failover step's deadline or the total deadline).
    finished: Option<(Duration, Result<sdk::ModelResponse, String>)>,
}

type AttemptLog = Arc<Mutex<Vec<Attempt>>>;

fn log_attempt(log: &AttemptLog, attempt: Attempt) -> usize {
    let mut attempts = log.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    attempts.push(attempt);
    attempts.len() - 1
}

/// A provider that notes each attempt in the log as it starts and as it
/// ends: the chain's steps, and the safety judge's backend.
struct LoggedProvider {
    inner: ModelBackend,
    log: AttemptLog,
}

impl LoggedProvider {
    fn start(&self, model: &str) -> (usize, Instant) {
        let started = Instant::now();
        let index = log_attempt(
            &self.log,
            Attempt {
                model: model.to_string(),
                started,
                finished: None,
            },
        );
        (index, started)
    }
}

fn finish(
    log: &AttemptLog,
    index: usize,
    started: Instant,
    outcome: Result<sdk::ModelResponse, String>,
) {
    if let Some(attempt) = log
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get_mut(index)
    {
        attempt.finished = Some((started.elapsed(), outcome));
    }
}

#[async_trait]
impl sdk::Provider for LoggedProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn complete(
        &self,
        request: &sdk::ModelRequest,
    ) -> Result<sdk::ModelResponse, sdk::ProviderError> {
        let (index, started) = self.start(&request.model);
        let result = self.inner.complete(request).await;
        let outcome = match &result {
            Ok(response) => Ok(response.clone()),
            Err(error) => Err(error.to_string()),
        };
        finish(&self.log, index, started, outcome);
        result
    }

    fn stream<'a>(
        &'a self,
        request: &'a sdk::ModelRequest,
    ) -> futures::stream::BoxStream<'a, Result<sdk::StreamEvent, sdk::ProviderError>> {
        use futures::StreamExt;
        let (index, started) = self.start(&request.model);
        let log = self.log.clone();
        Box::pin(
            self.inner
                .stream(request)
                .inspect(move |event| match event {
                    Ok(sdk::StreamEvent::Done(response)) => {
                        finish(&log, index, started, Ok(response.clone()))
                    }
                    Err(error) => finish(&log, index, started, Err(error.to_string())),
                    Ok(_) => {}
                }),
        )
    }
}

/// How much longer than the total deadline the safety judge's own budget
/// runs, so the total deadline always ends a hung judgment first and the
/// caller gets [`SideCallError::Deadline`], never the judge's own timeout.
const JUDGE_BUDGET_SLACK: Duration = Duration::from_secs(1);

impl SideCallGateway {
    pub fn new(
        runtime: SharedLlmRuntime,
        state: Arc<dyn StateStore>,
        event_store: Arc<EventStore>,
    ) -> Self {
        Self {
            runtime,
            state,
            event_store,
        }
    }

    /// One request, no tools. `model` is required when the spec says
    /// `ModelChoice::CallerSupplied` and ignored otherwise.
    #[allow(dead_code)] // used from Task 3
    pub async fn ask(
        &self,
        call: SideCall,
        messages: &[serde_json::Value],
        options: &sdk::RequestOptions,
        model: Option<&str>,
    ) -> Result<SideCallAnswer, SideCallError> {
        let spec = call.spec();
        let model = self.model_for(&spec, model).ok_or_else(|| {
            SideCallError::Provider(format!("no model supplied for {}", spec.purpose))
        })?;
        let request = sdk::ModelRequest {
            model,
            messages: to_sdk_messages(messages),
            options: options.clone(),
            ..Default::default()
        };
        let started = Instant::now();
        let (response, model) = self.run(spec, request).await?;
        if response.text.trim().is_empty() {
            return Err(SideCallError::Empty);
        }
        Ok(SideCallAnswer {
            text: response.text,
            model,
            elapsed: started.elapsed(),
        })
    }

    /// A provider for library code that makes the request itself. Each
    /// `complete` on it is bounded by the spec's deadline and recorded.
    #[allow(dead_code)] // used from Task 4
    pub fn provider(&self, call: SideCall) -> Arc<dyn aidaemon_core::Provider> {
        Arc::new(SideCallProvider {
            gateway: self.clone(),
            spec: call.spec(),
        })
    }

    /// The model the spec runs on; `supplied` is the caller's choice.
    fn model_for(&self, spec: &SideCallSpec, supplied: Option<&str>) -> Option<String> {
        let snapshot = self.runtime.snapshot();
        match spec.model {
            ModelChoice::StrongestFirst | ModelChoice::Primary => Some(snapshot.primary_model()),
            ModelChoice::Fast | ModelChoice::Cheap => Some(snapshot.cheap_model()),
            ModelChoice::CallerSupplied => supplied.map(str::to_string),
        }
    }

    /// Every attempt of one side call within its total deadline, then one
    /// row per attempt. Returns the response and the model that gave it.
    async fn run(
        &self,
        spec: SideCallSpec,
        mut request: sdk::ModelRequest,
    ) -> Result<(sdk::ModelResponse, String), SideCallError> {
        // No side call gets tools, whichever door the request came through.
        request.tools.clear();
        request.tool_choice = sdk::ToolChoice::default();
        let log = AttemptLog::default();
        let started = Instant::now();
        let result =
            match tokio::time::timeout(spec.deadline, self.attempt(&spec, &request, &log)).await {
                Ok(result) => result,
                Err(_) => Err(SideCallError::Deadline(spec.deadline)),
            };
        self.record(&spec, &request.model, &log, &result, started.elapsed())
            .await;
        result
    }

    async fn attempt(
        &self,
        spec: &SideCallSpec,
        request: &sdk::ModelRequest,
        log: &AttemptLog,
    ) -> Result<(sdk::ModelResponse, String), SideCallError> {
        let snapshot = self.runtime.snapshot();
        let logged = |inner| -> ModelBackend {
            Arc::new(LoggedProvider {
                inner,
                log: log.clone(),
            })
        };
        if spec.class == SideCallClass::SafetyJudge {
            // Each attempt is logged as it starts, so the rows survive the
            // total deadline cutting the judgment off.
            let outcome = snapshot
                .ask_for_judgment_through(
                    &to_chat_messages(&request.messages),
                    &to_chat_options(request),
                    spec.deadline + JUDGE_BUDGET_SLACK,
                    logged,
                )
                .await;
            let Some(answer) = outcome.answer else {
                return Err(SideCallError::Provider(
                    outcome
                        .failures
                        .last()
                        .map(|failure| failure.error.clone())
                        .unwrap_or_else(|| "no model available".to_string()),
                ));
            };
            return Ok((to_sdk_response(answer.response), answer.model));
        }
        let chain = snapshot.background_chain_through(&request.model, logged);
        let response = sdk::Provider::complete(&chain, request)
            .await
            .map_err(|error| SideCallError::Provider(error.to_string()))?;
        let model = log
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .rev()
            .find(|attempt| matches!(attempt.finished, Some((_, Ok(_)))))
            .map(|attempt| attempt.model.clone())
            .unwrap_or_else(|| request.model.clone());
        Ok((response, model))
    }

    /// One `llm_call` row per attempt, under the session
    /// `background:<purpose>`. An attempt still open when the call ended
    /// is a failed row; a failed call with no attempt at all gets one
    /// failed row for its model, so no failure goes unrecorded. Recording
    /// errors are logged by the recorder and never fail the call.
    async fn record(
        &self,
        spec: &SideCallSpec,
        model: &str,
        log: &AttemptLog,
        result: &Result<(sdk::ModelResponse, String), SideCallError>,
        elapsed: Duration,
    ) {
        let attempts =
            std::mem::take(&mut *log.lock().unwrap_or_else(|poisoned| poisoned.into_inner()));
        let session = format!("background:{}", spec.purpose);
        let unfinished = match result {
            Err(error @ SideCallError::Deadline(_)) => error.to_string(),
            _ => "abandoned without an answer at its step deadline".to_string(),
        };
        if attempts.is_empty() {
            if let Err(error) = result {
                self.record_failure(&session, spec, model, &error.to_string(), elapsed)
                    .await;
            }
            return;
        }
        for attempt in attempts {
            match attempt.finished {
                Some((elapsed, Ok(response))) => {
                    crate::events::record_background_model_call_telemetry(
                        self.event_store.clone(),
                        self.state.as_ref(),
                        &session,
                        spec.purpose,
                        &attempt.model,
                        &to_provider_response(response, &attempt.model),
                        elapsed,
                    )
                    .await;
                }
                Some((elapsed, Err(error))) => {
                    self.record_failure(&session, spec, &attempt.model, &error, elapsed)
                        .await;
                }
                None => {
                    self.record_failure(
                        &session,
                        spec,
                        &attempt.model,
                        &unfinished,
                        attempt.started.elapsed(),
                    )
                    .await;
                }
            }
        }
    }

    async fn record_failure(
        &self,
        session: &str,
        spec: &SideCallSpec,
        model: &str,
        error: &str,
        elapsed: Duration,
    ) {
        tracing::warn!(purpose = spec.purpose, %model, %error, "side call attempt failed");
        crate::events::record_background_model_call_failure(
            self.event_store.clone(),
            self.state.as_ref(),
            session,
            spec.purpose,
            model,
            error,
            elapsed,
        )
        .await;
    }
}

/// See [`SideCallGateway::provider`].
struct SideCallProvider {
    gateway: SideCallGateway,
    spec: SideCallSpec,
}

#[async_trait]
impl sdk::Provider for SideCallProvider {
    fn name(&self) -> &str {
        "side-call"
    }

    /// The request's own model when the spec says the caller supplies it,
    /// else the spec's model.
    async fn complete(
        &self,
        request: &sdk::ModelRequest,
    ) -> Result<sdk::ModelResponse, sdk::ProviderError> {
        let mut request = request.clone();
        if let Some(model) = self.gateway.model_for(&self.spec, Some(&request.model)) {
            request.model = model;
        }
        self.gateway
            .run(self.spec, request)
            .await
            .map(|(response, _)| response)
            .map_err(|error| sdk::ProviderError::Other(error.to_string()))
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! Gateways over scripted providers, and counts of the rows they write.
    //! Shared with the tests of the modules whose side calls move here.

    use std::sync::Arc;

    use aidaemon_core as sdk;
    use async_trait::async_trait;

    use super::SideCallGateway;
    use crate::config::ProviderKind;
    use crate::events::{EventStore, EventStoreForTests as _};
    use crate::llm_runtime::{ModelBackend, ProviderRuntimeTarget, SharedLlmRuntime};
    use crate::traits::StateStore;

    /// A provider that answers every request with `text`.
    pub(crate) fn provider_answering(text: &str) -> ModelBackend {
        Arc::new(
            aidaemon_core::testing::ScriptedProvider::new([])
                .then_repeat(aidaemon_core::testing::text(text)),
        )
    }

    struct Failing(sdk::ProviderError);

    #[async_trait]
    impl sdk::Provider for Failing {
        fn name(&self) -> &str {
            "failing"
        }

        async fn complete(
            &self,
            _: &sdk::ModelRequest,
        ) -> Result<sdk::ModelResponse, sdk::ProviderError> {
            Err(self.0.clone())
        }
    }

    /// A provider that fails every request with a transient `error`.
    pub(crate) fn provider_failing(error: &str) -> ModelBackend {
        provider_failing_with(sdk::ProviderError::Transient(error.to_string()))
    }

    /// A provider that fails every request with `error`.
    pub(crate) fn provider_failing_with(error: sdk::ProviderError) -> ModelBackend {
        Arc::new(Failing(error))
    }

    struct Hanging;

    #[async_trait]
    impl sdk::Provider for Hanging {
        fn name(&self) -> &str {
            "hanging"
        }

        async fn complete(
            &self,
            _: &sdk::ModelRequest,
        ) -> Result<sdk::ModelResponse, sdk::ProviderError> {
            std::future::pending().await
        }
    }

    /// A provider that never answers.
    pub(crate) fn provider_hanging() -> ModelBackend {
        Arc::new(Hanging)
    }

    async fn state() -> Arc<crate::state::SqliteStateStore> {
        let file = tempfile::NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap().to_string();
        // The store keeps the file open; the file outlives the test.
        std::mem::forget(file);
        let embeddings = Arc::new(crate::memory::embeddings::EmbeddingService::new().unwrap());
        Arc::new(
            crate::state::SqliteStateStore::new(&path, 100, None, embeddings)
                .await
                .unwrap(),
        )
    }

    /// A gateway whose primary provider is the first target and whose
    /// failover providers are the rest, each with its own model.
    pub(crate) async fn gateway_with_targets(
        targets: Vec<ModelBackend>,
    ) -> (SideCallGateway, Arc<EventStore>) {
        let state = state().await;
        let events = Arc::new(EventStore::new(state.pool()).await.unwrap());
        let state: Arc<dyn StateStore> = state;
        (
            SideCallGateway::new(runtime(targets), state, events.clone()),
            events,
        )
    }

    /// A gateway over one provider.
    pub(crate) async fn gateway_with(provider: ModelBackend) -> (SideCallGateway, Arc<EventStore>) {
        gateway_with_targets(vec![provider]).await
    }

    /// A gateway whose event store cannot be written.
    pub(crate) async fn gateway_with_broken_event_store(
        provider: ModelBackend,
    ) -> (SideCallGateway, Arc<EventStore>) {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        let events = Arc::new(EventStore::new(pool.clone()).await.unwrap());
        pool.close().await;
        let state: Arc<dyn StateStore> = state().await;
        (
            SideCallGateway::new(runtime(vec![provider]), state, events.clone()),
            events,
        )
    }

    fn runtime(targets: Vec<ModelBackend>) -> SharedLlmRuntime {
        let mut targets = targets.into_iter();
        let primary = targets.next().expect("at least one provider");
        let failovers = targets
            .enumerate()
            .map(|(n, provider)| {
                ProviderRuntimeTarget::new(
                    provider,
                    None,
                    ProviderKind::Anthropic,
                    format!("failover-model-{}", n + 1),
                )
            })
            .collect();
        SharedLlmRuntime::new_with_failovers(
            primary,
            None,
            ProviderKind::OpenaiCompatible,
            "primary-model".to_string(),
            failovers,
        )
    }

    async fn count_calls(events: &EventStore, purpose: &str, failed: bool) -> usize {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT data FROM events WHERE event_type = ? \
             AND json_extract(data, '$.call_purpose') = ?",
        )
        .bind(crate::events::EventType::LlmCall.as_str())
        .bind(purpose)
        .fetch_all(&events.pool())
        .await
        .unwrap();
        rows.iter()
            .filter(|(data,)| {
                let data: serde_json::Value = serde_json::from_str(data).unwrap();
                data.get("error").is_some_and(|error| !error.is_null()) == failed
            })
            .count()
    }

    /// The models of the failed `llm_call` rows recorded under `purpose`,
    /// in the order they were written.
    pub(crate) async fn failed_call_models(events: &EventStore, purpose: &str) -> Vec<String> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT data FROM events WHERE event_type = ? \
             AND json_extract(data, '$.call_purpose') = ? \
             AND json_extract(data, '$.error') IS NOT NULL ORDER BY id",
        )
        .bind(crate::events::EventType::LlmCall.as_str())
        .bind(purpose)
        .fetch_all(&events.pool())
        .await
        .unwrap();
        rows.iter()
            .map(|(data,)| {
                let data: serde_json::Value = serde_json::from_str(data).unwrap();
                data["model"].as_str().unwrap_or_default().to_string()
            })
            .collect()
    }

    /// `llm_call` rows recorded under `purpose` that succeeded.
    pub(crate) async fn ok_calls(events: &EventStore, purpose: &str) -> usize {
        count_calls(events, purpose, false).await
    }

    /// `llm_call` rows recorded under `purpose` that failed.
    pub(crate) async fn failed_calls(events: &EventStore, purpose: &str) -> usize {
        count_calls(events, purpose, true).await
    }

    /// One user message, as chat JSON.
    pub(crate) fn user(text: &str) -> Vec<serde_json::Value> {
        vec![serde_json::json!({"role": "user", "content": text})]
    }

    /// A request for `model` with one user message.
    pub(crate) fn request_for(model: &str, text: &str) -> sdk::ModelRequest {
        sdk::ModelRequest {
            model: model.to_string(),
            messages: vec![sdk::Message::user(text)],
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::test_support::*;
    use super::*;

    #[tokio::test]
    async fn a_success_is_recorded_under_the_spec_purpose() {
        let (gateway, events) = gateway_with(provider_answering("fine")).await;
        let answer = gateway
            .ask(
                SideCall::BackgroundInterpretation,
                &user("hello"),
                &Default::default(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(answer.text, "fine");
        assert_eq!(ok_calls(&events, "background_interpretation").await, 1);
    }

    #[tokio::test]
    async fn a_failure_is_recorded_too() {
        let (gateway, events) = gateway_with(provider_failing("503")).await;
        let result = gateway
            .ask(
                SideCall::BackgroundInterpretation,
                &user("hello"),
                &Default::default(),
                None,
            )
            .await;
        assert!(matches!(result, Err(SideCallError::Provider(_))));
        assert!(failed_calls(&events, "background_interpretation").await >= 1);
        assert_eq!(ok_calls(&events, "background_interpretation").await, 0);
    }

    #[tokio::test]
    async fn a_hanging_provider_ends_at_the_total_deadline() {
        // Two failover targets, both hang forever.
        let (gateway, events) =
            gateway_with_targets(vec![provider_hanging(), provider_hanging()]).await;
        // The wait runs on virtual time. The database pool times out under
        // virtual time, so real time resumes the moment the deadline passes,
        // before the gateway writes its row.
        tokio::time::pause();
        let deadline = SideCall::BackgroundInterpretation.spec().deadline;
        let started = tokio::time::Instant::now();
        let messages = user("hello");
        let options = Default::default();
        let (result, ()) = tokio::join!(
            gateway.ask(
                SideCall::BackgroundInterpretation,
                &messages,
                &options,
                None
            ),
            async {
                tokio::time::sleep(deadline).await;
                tokio::time::resume();
            }
        );
        assert!(matches!(result, Err(SideCallError::Deadline(_))));
        assert!(started.elapsed() <= Duration::from_secs(31));
        assert!(failed_calls(&events, "background_interpretation").await >= 1);
    }

    #[tokio::test]
    async fn a_recording_failure_does_not_fail_the_call() {
        let (gateway, _events) = gateway_with_broken_event_store(provider_answering("fine")).await;
        let answer = gateway
            .ask(
                SideCall::BackgroundInterpretation,
                &user("hello"),
                &Default::default(),
                None,
            )
            .await
            .unwrap();
        assert_eq!(answer.text, "fine");
    }

    #[tokio::test]
    async fn an_empty_reply_is_an_error() {
        let (gateway, _events) = gateway_with(provider_answering("   ")).await;
        let result = gateway
            .ask(
                SideCall::BackgroundInterpretation,
                &user("hello"),
                &Default::default(),
                None,
            )
            .await;
        assert!(matches!(result, Err(SideCallError::Empty)));
    }

    #[tokio::test]
    async fn the_provider_handle_records_like_ask() {
        let (gateway, events) = gateway_with(provider_answering("fine")).await;
        let provider = gateway.provider(SideCall::TaskLearning);
        provider
            .complete(&request_for("any-model", "hello"))
            .await
            .unwrap();
        assert_eq!(ok_calls(&events, "task_learning").await, 1);
    }

    /// A safety judge whose primary is rejected and whose failover model
    /// hangs: the total deadline ends it, and each attempt has its row
    /// under the model it used.
    #[tokio::test]
    async fn a_judge_cut_off_by_the_deadline_records_each_attempt() {
        let (gateway, events) = gateway_with_targets(vec![
            provider_failing_with(sdk::ProviderError::Auth("key rejected".into())),
            provider_hanging(),
        ])
        .await;
        // Virtual time for the wait, real time again for the row writes
        // (see `a_hanging_provider_ends_at_the_total_deadline`).
        tokio::time::pause();
        let deadline = SideCall::CommandRisk.spec().deadline;
        let started = tokio::time::Instant::now();
        let messages = user("rm -rf build");
        let options = Default::default();
        let (result, ()) = tokio::join!(
            gateway.ask(SideCall::CommandRisk, &messages, &options, None),
            async {
                tokio::time::sleep(deadline).await;
                tokio::time::resume();
            }
        );
        assert!(
            matches!(result, Err(SideCallError::Deadline(d)) if d == deadline),
            "{result:?}"
        );
        assert!(started.elapsed() <= deadline + Duration::from_secs(1));
        assert_eq!(
            failed_call_models(&events, "terminal_command_risk").await,
            ["primary-model", "failover-model-1"]
        );
        assert_eq!(ok_calls(&events, "terminal_command_risk").await, 0);
    }

    #[tokio::test]
    async fn a_side_call_never_sends_tools() {
        let scripted = Arc::new(
            aidaemon_core::testing::ScriptedProvider::new([])
                .then_repeat(aidaemon_core::testing::text("fine")),
        );
        let (gateway, _events) = gateway_with(scripted.clone()).await;
        let mut request = request_for("any-model", "hello");
        request.tools = vec![sdk::ToolSpec {
            name: "send_probe".to_string(),
            description: "Sends a message.".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        }];
        request.tool_choice = sdk::ToolChoice::Required;
        gateway
            .provider(SideCall::TaskLearning)
            .complete(&request)
            .await
            .unwrap();
        let seen = scripted.requests();
        assert_eq!(seen.len(), 1);
        assert!(seen[0].tools.is_empty());
        assert_eq!(seen[0].tool_choice, sdk::ToolChoice::Auto);
    }
}
