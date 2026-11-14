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
use crate::chat_bridge::sdk_backed::{
    to_chat_messages, to_chat_options, to_provider_response, to_sdk_messages, to_sdk_response,
};
use aidaemon_events::EventStore;
use aidaemon_events::provider::TokenUsageStore;

/// Runs side calls: picks the model chain from the spec, applies its total
/// deadline and records every attempt.
#[derive(Clone)]
pub struct SideCallGateway {
    runtime: SharedLlmRuntime,
    state: Arc<dyn TokenUsageStore>,
    event_store: Arc<EventStore>,
    /// The optional decision backend (`decide`); empty when none is set.
    pub(super) decisions: Arc<std::sync::RwLock<Option<super::decide::DecisionLayer>>>,
}

/// A side call's answer.
#[derive(Debug, Clone)]
pub struct SideCallAnswer {
    pub text: String,
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

    fn unavailable_until(&self, model: &str) -> Option<std::time::SystemTime> {
        self.inner.unavailable_until(model)
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

/// How long the gateway waits for a side call's telemetry rows after the
/// model call ended, so a stuck event store never holds the caller.
const RECORD_DEADLINE: Duration = Duration::from_secs(2);

impl SideCallGateway {
    pub fn new(
        runtime: SharedLlmRuntime,
        state: Arc<dyn TokenUsageStore>,
        event_store: Arc<EventStore>,
    ) -> Self {
        Self {
            runtime,
            state,
            event_store,
            decisions: Default::default(),
        }
    }

    pub(super) fn runtime_handle(&self) -> &SharedLlmRuntime {
        &self.runtime
    }

    pub(super) fn event_store_handle(&self) -> Arc<EventStore> {
        self.event_store.clone()
    }

    pub(super) fn state_handle(&self) -> Arc<dyn TokenUsageStore> {
        self.state.clone()
    }

    /// One request, no tools. `model` is required when the spec says
    /// `ModelChoice::CallerSupplied` and ignored otherwise.
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
        let response = self.run(spec, request).await?;
        if response.text.trim().is_empty() {
            return Err(SideCallError::Empty);
        }
        Ok(SideCallAnswer {
            text: response.text,
        })
    }

    /// A provider for library code that makes the request itself. Each
    /// `complete` on it is bounded by the spec's deadline and recorded.
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
            ModelChoice::Primary => Some(snapshot.primary_model()),
            ModelChoice::Cheap => Some(snapshot.cheap_model()),
            ModelChoice::CallerSupplied => supplied.map(str::to_string),
        }
    }

    /// Every attempt of one side call within its total deadline, then one
    /// row per attempt.
    async fn run(
        &self,
        spec: SideCallSpec,
        mut request: sdk::ModelRequest,
    ) -> Result<sdk::ModelResponse, SideCallError> {
        // No side call runs tools, whichever door the request came through.
        // A forced call to one named tool stays: it is how a caller asks for
        // structured output, and the call only comes back to the caller.
        // Otherwise `None`, not the default `Auto`: with no tools, `Auto`
        // lets a backend add its own (Google turns on search grounding).
        match &request.tool_choice {
            sdk::ToolChoice::Specific(name) => request.tools.retain(|tool| tool.name == *name),
            _ => request.tools.clear(),
        }
        if request.tools.is_empty() {
            request.tool_choice = sdk::ToolChoice::None;
        }
        let log = AttemptLog::default();
        // A caller that drops this future (its own timeout, a cancelled
        // `select!` arm) still gets a row for every attempt.
        let guard = DropRecorder {
            gateway: self.clone(),
            spec,
            log: log.clone(),
            armed: true,
        };
        let started = Instant::now();
        let result =
            match tokio::time::timeout(spec.deadline, self.attempt(&spec, &request, &log)).await {
                Ok(result) => result,
                Err(_) => Err(SideCallError::Deadline(spec.deadline)),
            };
        let elapsed = started.elapsed();
        // One line per side call, so a turn's latency splits by grep alone.
        tracing::info!(
            purpose = spec.purpose,
            model = %request.model,
            elapsed_ms = elapsed.as_millis() as u64,
            outcome = match &result {
                Ok(_) => "answered",
                Err(SideCallError::Deadline(_)) => "deadline",
                Err(SideCallError::Provider(_)) => "failed",
                Err(SideCallError::Empty) => "empty",
            },
            "Side call finished"
        );
        // An attempt leaves the log before its row is written, so a write
        // cut off here is not written again by the guard.
        if tokio::time::timeout(
            RECORD_DEADLINE,
            self.record(&spec, &request.model, &log, &result, elapsed),
        )
        .await
        .is_err()
        {
            tracing::warn!(
                purpose = spec.purpose,
                "side call telemetry write timed out; rows not written"
            );
        }
        guard.disarm();
        result
    }

    async fn attempt(
        &self,
        spec: &SideCallSpec,
        request: &sdk::ModelRequest,
        log: &AttemptLog,
    ) -> Result<sdk::ModelResponse, SideCallError> {
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
            return snapshot
                .ask_for_judgment_through(
                    &to_chat_messages(&request.messages),
                    &to_chat_options(request),
                    spec.deadline + JUDGE_BUDGET_SLACK,
                    logged,
                )
                .await
                .map(to_sdk_response)
                .map_err(SideCallError::Provider);
        }
        let chain = snapshot.background_chain_through(&request.model, logged);
        sdk::Provider::complete(&chain, request)
            .await
            .map_err(|error| SideCallError::Provider(error.to_string()))
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
        result: &Result<sdk::ModelResponse, SideCallError>,
        elapsed: Duration,
    ) {
        let unfinished = match result {
            Err(error @ SideCallError::Deadline(_)) => error.to_string(),
            _ => "abandoned without an answer at its step deadline".to_string(),
        };
        let no_attempts = log
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_empty();
        if no_attempts {
            if let Err(error) = result {
                self.record_failure(spec, model, &error.to_string(), elapsed)
                    .await;
            }
            return;
        }
        self.record_attempts(spec, log, &unfinished).await;
    }

    /// Writes and removes the logged attempts one at a time, so an attempt
    /// is written once even if the caller drops the call mid-way (the drop
    /// guard writes what is left).
    async fn record_attempts(&self, spec: &SideCallSpec, log: &AttemptLog, unfinished: &str) {
        let session = format!("background:{}", spec.purpose);
        loop {
            let attempt = {
                let mut attempts = log.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                if attempts.is_empty() {
                    break;
                }
                attempts.remove(0)
            };
            match attempt.finished {
                Some((elapsed, Ok(response))) => {
                    aidaemon_events::model_call_telemetry::record_background_model_call_telemetry(
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
                    self.record_failure(spec, &attempt.model, &error, elapsed)
                        .await;
                }
                None => {
                    self.record_failure(
                        spec,
                        &attempt.model,
                        unfinished,
                        attempt.started.elapsed(),
                    )
                    .await;
                }
            }
        }
    }

    async fn record_failure(
        &self,
        spec: &SideCallSpec,
        model: &str,
        error: &str,
        elapsed: Duration,
    ) {
        tracing::warn!(purpose = spec.purpose, %model, %error, "side call attempt failed");
        aidaemon_events::model_call_telemetry::record_background_model_call_failure(
            self.event_store.clone(),
            self.state.as_ref(),
            &format!("background:{}", spec.purpose),
            spec.purpose,
            model,
            error,
            elapsed,
        )
        .await;
    }
}

/// What an attempt that never ended is recorded as when the caller drops
/// the side call.
const DROPPED_BY_CALLER: &str = "dropped by the caller before an answer";

/// Records the attempts of a side call whose caller dropped it before the
/// gateway wrote its rows. Disarmed once the rows are written.
struct DropRecorder {
    gateway: SideCallGateway,
    spec: SideCallSpec,
    log: AttemptLog,
    armed: bool,
}

impl DropRecorder {
    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for DropRecorder {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let pending = !self
            .log
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_empty();
        if !pending {
            return;
        }
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::warn!(
                purpose = self.spec.purpose,
                "side call dropped outside a runtime; its attempts go unrecorded"
            );
            return;
        };
        let gateway = self.gateway.clone();
        let spec = self.spec;
        let log = self.log.clone();
        handle.spawn(async move {
            gateway
                .record_attempts(&spec, &log, DROPPED_BY_CALLER)
                .await;
        });
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
            .map_err(|error| sdk::ProviderError::Other(error.to_string()))
    }
}

/// Skill promotion writes its skill through the gateway's skill-promotion
/// side call.
#[async_trait::async_trait]
impl aidaemon_skills::promotion::PromotionModel for SideCallGateway {
    async fn write_skill(&self, messages: &[serde_json::Value]) -> anyhow::Result<String> {
        Ok(self
            .ask(
                crate::llm::side_calls::SideCall::SkillPromotion,
                messages,
                &Default::default(),
                None,
            )
            .await?
            .text)
    }
}

#[cfg(any(test, feature = "testing"))]
pub mod test_support {
    //! Gateways over scripted providers, and counts of the rows they write.
    //! Shared with the tests of the modules whose side calls move here.

    use std::sync::Arc;

    use aidaemon_core as sdk;
    use async_trait::async_trait;

    use super::SideCallGateway;
    use crate::llm::{ModelBackend, ProviderRuntimeTarget, SharedLlmRuntime};
    use aidaemon_events::EventStore;
    use aidaemon_events::provider::TokenUsageStore;
    use aidaemon_providers::routing::ProviderKind;

    /// A provider that answers every request with `text`.
    pub fn provider_answering(text: &str) -> ModelBackend {
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
    pub fn provider_failing(error: &str) -> ModelBackend {
        provider_failing_with(sdk::ProviderError::Transient(error.to_string()))
    }

    /// A provider that fails every request with `error`.
    pub fn provider_failing_with(error: sdk::ProviderError) -> ModelBackend {
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
    pub fn provider_hanging() -> ModelBackend {
        Arc::new(Hanging)
    }

    /// Token usage goes through the event store; the side-call path never
    /// reads the usage store.
    struct NoUsage;

    #[async_trait]
    impl TokenUsageStore for NoUsage {}

    fn state() -> Arc<dyn TokenUsageStore> {
        Arc::new(NoUsage)
    }

    async fn new_event_store(pool: sqlx::SqlitePool) -> EventStore {
        EventStore::on_paths(
            pool,
            aidaemon_tools::LexicalPaths::daemon_host(),
            Arc::new(aidaemon_events::NoEventHost),
        )
        .await
        .unwrap()
    }

    /// An event store on a fresh file, with the host's `token_usage` table.
    async fn event_store() -> Arc<EventStore> {
        let file = tempfile::NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap().to_string();
        // The pool keeps the file open; the file outlives the test.
        std::mem::forget(file);
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .connect_with(
                sqlx::sqlite::SqliteConnectOptions::new()
                    .filename(&path)
                    .create_if_missing(true),
            )
            .await
            .unwrap();
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cached_input_tokens INTEGER,
                cache_creation_input_tokens INTEGER,
                call_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        Arc::new(new_event_store(pool).await)
    }

    /// A gateway whose primary provider is the first target and whose
    /// failover providers are the rest, each with its own model.
    pub async fn gateway_with_targets(
        targets: Vec<ModelBackend>,
    ) -> (SideCallGateway, Arc<EventStore>) {
        let events = event_store().await;
        (
            SideCallGateway::new(runtime(targets), state(), events.clone()),
            events,
        )
    }

    /// A gateway over one provider.
    pub async fn gateway_with(provider: ModelBackend) -> (SideCallGateway, Arc<EventStore>) {
        gateway_with_targets(vec![provider]).await
    }

    /// A gateway whose event store cannot be written.
    pub async fn gateway_with_broken_event_store(
        provider: ModelBackend,
    ) -> (SideCallGateway, Arc<EventStore>) {
        let pool = sqlx::SqlitePool::connect("sqlite::memory:").await.unwrap();
        let events = Arc::new(new_event_store(pool.clone()).await);
        pool.close().await;
        let state = state();
        (
            SideCallGateway::new(runtime(vec![provider]), state, events.clone()),
            events,
        )
    }

    /// A gateway whose event store writes wait forever: its one pooled
    /// connection is held by the returned handle. Keep the handle alive.
    pub async fn gateway_with_stuck_event_store(
        provider: ModelBackend,
    ) -> (SideCallGateway, sqlx::pool::PoolConnection<sqlx::Sqlite>) {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .acquire_timeout(std::time::Duration::from_secs(3600))
            .connect("sqlite::memory:")
            .await
            .unwrap();
        let events = Arc::new(new_event_store(pool.clone()).await);
        let held = pool.acquire().await.unwrap();
        let state = state();
        (
            SideCallGateway::new(runtime(vec![provider]), state, events),
            held,
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
        .bind(aidaemon_events::EventType::LlmCall.as_str())
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
    pub async fn failed_call_models(events: &EventStore, purpose: &str) -> Vec<String> {
        let rows: Vec<(String,)> = sqlx::query_as(
            "SELECT data FROM events WHERE event_type = ? \
             AND json_extract(data, '$.call_purpose') = ? \
             AND json_extract(data, '$.error') IS NOT NULL ORDER BY id",
        )
        .bind(aidaemon_events::EventType::LlmCall.as_str())
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
    pub async fn ok_calls(events: &EventStore, purpose: &str) -> usize {
        count_calls(events, purpose, false).await
    }

    /// `llm_call` rows recorded under `purpose` that failed.
    pub async fn failed_calls(events: &EventStore, purpose: &str) -> usize {
        count_calls(events, purpose, true).await
    }

    /// One user message, as chat JSON.
    pub fn user(text: &str) -> Vec<serde_json::Value> {
        vec![serde_json::json!({"role": "user", "content": text})]
    }

    /// A request for `model` with one user message.
    pub fn request_for(model: &str, text: &str) -> sdk::ModelRequest {
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

    /// A telemetry write that never finishes does not hold the caller: the
    /// answer comes back once the write's own deadline passes.
    #[tokio::test]
    async fn a_stuck_telemetry_write_does_not_hold_the_answer() {
        let (gateway, _held) = gateway_with_stuck_event_store(provider_answering("fine")).await;
        tokio::time::pause();
        let deadline = SideCall::BackgroundInterpretation.spec().deadline;
        let started = tokio::time::Instant::now();
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
        assert!(started.elapsed() <= deadline + RECORD_DEADLINE);
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
    /// Live 2026-11-12: with the ChatGPT plan used up, every safety
    /// judgment still tried the primary model first. After one refusal
    /// stating its reset, the model is skipped until then.
    #[tokio::test]
    async fn a_judgment_skips_a_model_out_of_quota_until_its_reset() {
        let reset = std::time::SystemTime::now() + Duration::from_secs(3600);
        let capped: ModelBackend = Arc::new(sdk::QuotaGuard::from_arc(provider_failing_with(
            sdk::ProviderError::QuotaExhausted {
                message: "ChatGPT usage limit reached (plus plan)".into(),
                resets_at: Some(reset),
                per_model: false,
                in_stream: false,
            },
        )));
        let verdict = r#"{"consistent":true,"reason":"asked","owner_reason":"You asked."}"#;
        let (gateway, events) =
            gateway_with_targets(vec![capped, provider_answering(verdict)]).await;
        for _ in 0..3 {
            let answer = gateway
                .ask(
                    SideCall::UntrustedVerifier,
                    &user("judge"),
                    &Default::default(),
                    None,
                )
                .await
                .expect("the failover judges");
            assert!(answer.text.contains("consistent"));
        }
        assert_eq!(
            failed_call_models(&events, "untrusted_call_verifier").await,
            ["primary-model"],
            "the used-up model is tried once"
        );
    }

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
        assert_eq!(seen[0].tool_choice, sdk::ToolChoice::None);
    }

    fn tool_named(name: &str) -> sdk::ToolSpec {
        sdk::ToolSpec {
            name: name.to_string(),
            description: "Records something.".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        }
    }

    /// A forced call to one named tool is how a caller asks for structured
    /// output: that tool stays, every other is removed, and the call comes
    /// back to the caller with nothing having run it.
    #[tokio::test]
    async fn a_forced_tool_call_keeps_only_that_tool() {
        let scripted = Arc::new(
            aidaemon_core::testing::ScriptedProvider::new([]).then_repeat(
                aidaemon_core::testing::tool_calls([aidaemon_core::testing::call(
                    "call-1",
                    "record_facts",
                    serde_json::json!({"facts": []}),
                )]),
            ),
        );
        let (gateway, _events) = gateway_with(scripted.clone()).await;
        let mut request = request_for("any-model", "hello");
        request.tools = vec![tool_named("send_probe"), tool_named("record_facts")];
        request.tool_choice = sdk::ToolChoice::Specific("record_facts".to_string());
        let response = gateway
            .provider(SideCall::FactExtraction)
            .complete(&request)
            .await
            .unwrap();
        let seen = scripted.requests();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].tools.len(), 1);
        assert_eq!(seen[0].tools[0].name, "record_facts");
        assert_eq!(
            seen[0].tool_choice,
            sdk::ToolChoice::Specific("record_facts".to_string())
        );
        assert_eq!(response.tool_calls[0].name, "record_facts");
    }

    /// Forcing a tool the request does not define leaves nothing to call.
    #[tokio::test]
    async fn forcing_an_undefined_tool_sends_no_tools() {
        let scripted = Arc::new(
            aidaemon_core::testing::ScriptedProvider::new([])
                .then_repeat(aidaemon_core::testing::text("fine")),
        );
        let (gateway, _events) = gateway_with(scripted.clone()).await;
        let mut request = request_for("any-model", "hello");
        request.tools = vec![tool_named("send_probe")];
        request.tool_choice = sdk::ToolChoice::Specific("record_facts".to_string());
        gateway
            .provider(SideCall::FactExtraction)
            .complete(&request)
            .await
            .unwrap();
        let seen = scripted.requests();
        assert!(seen[0].tools.is_empty());
        assert_eq!(seen[0].tool_choice, sdk::ToolChoice::None);
    }

    /// The judge path sends no tools and `ToolChoice::None` too.
    #[tokio::test]
    async fn a_safety_judge_never_sends_tools() {
        let scripted = Arc::new(
            aidaemon_core::testing::ScriptedProvider::new([])
                .then_repeat(aidaemon_core::testing::text("fine")),
        );
        let (gateway, _events) = gateway_with(scripted.clone()).await;
        gateway
            .ask(
                SideCall::CommandRisk,
                &user("rm -rf build"),
                &Default::default(),
                None,
            )
            .await
            .unwrap();
        let seen = scripted.requests();
        assert_eq!(seen.len(), 1);
        assert!(seen[0].tools.is_empty());
        assert_eq!(seen[0].tool_choice, sdk::ToolChoice::None);
    }

    /// Waits (real time) until `purpose` has `want` failed rows.
    async fn wait_for_failed_rows(events: &EventStore, purpose: &str, want: usize) -> usize {
        for _ in 0..200 {
            let found = failed_calls(events, purpose).await;
            if found >= want {
                return found;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        failed_calls(events, purpose).await
    }

    /// A caller's own timeout drops the call before the gateway's deadline;
    /// the open attempt still gets its failed row, written once.
    #[tokio::test]
    async fn a_call_dropped_by_its_caller_is_recorded_once() {
        for call in [SideCall::BackgroundInterpretation, SideCall::CommandRisk] {
            let (gateway, events) = gateway_with(provider_hanging()).await;
            let purpose = call.spec().purpose;
            let dropped = tokio::time::timeout(
                Duration::from_millis(200),
                gateway.ask(call, &user("hello"), &Default::default(), None),
            )
            .await;
            assert!(
                dropped.is_err(),
                "{purpose}: the caller's timeout fires first"
            );
            assert_eq!(
                wait_for_failed_rows(&events, purpose, 1).await,
                1,
                "{purpose}"
            );
            // Nothing else lands later.
            tokio::time::sleep(Duration::from_millis(100)).await;
            assert_eq!(failed_calls(&events, purpose).await, 1, "{purpose}");
            assert_eq!(ok_calls(&events, purpose).await, 0, "{purpose}");
            let models = failed_call_models(&events, purpose).await;
            assert_eq!(models, ["primary-model"], "{purpose}");
        }
    }

    /// A call that completes writes its one ok row and nothing more.
    #[tokio::test]
    async fn a_completed_call_is_not_recorded_again_on_drop() {
        let (gateway, events) = gateway_with(provider_answering("fine")).await;
        gateway
            .ask(
                SideCall::BackgroundInterpretation,
                &user("hello"),
                &Default::default(),
                None,
            )
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(ok_calls(&events, "background_interpretation").await, 1);
        assert_eq!(failed_calls(&events, "background_interpretation").await, 0);
    }
}
