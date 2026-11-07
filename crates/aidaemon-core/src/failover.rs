//! [`Failover`]: a chain of providers, each attempt with its own deadline.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::channel::mpsc;
use futures::stream::{self, BoxStream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::provider::{ModelRequest, ModelResponse, Provider, ProviderError, StreamEvent};

/// Longest wait between attempts on the same step, whatever a rate limit
/// asks for.
const MAX_BACKOFF: Duration = Duration::from_secs(30);

/// One link of a [`Failover`] chain: a provider, the model it is asked for,
/// how long an attempt may take to start answering, and how many attempts
/// it gets before the chain moves on.
#[derive(Clone)]
pub struct FailoverStep {
    provider: Arc<dyn Provider>,
    model: Option<String>,
    deadline: Duration,
    attempts: u32,
}

impl FailoverStep {
    pub fn new(provider: impl Provider + 'static) -> Self {
        Self::from_arc(Arc::new(provider))
    }

    pub fn from_arc(provider: Arc<dyn Provider>) -> Self {
        Self {
            provider,
            model: None,
            deadline: Duration::from_secs(90),
            attempts: 1,
        }
    }

    /// Ask this step for `model` instead of the request's model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// How long one attempt may take to produce its first stream event (or,
    /// through [`Provider::complete`], its whole response). Default 90s.
    /// Each attempt gets the full deadline, whatever earlier ones used.
    pub fn deadline(mut self, deadline: Duration) -> Self {
        self.deadline = deadline;
        self
    }

    /// Attempts on this step before the chain moves on (default 1).
    pub fn attempts(mut self, attempts: u32) -> Self {
        self.attempts = attempts.max(1);
        self
    }

    fn label(&self, request: &ModelRequest) -> String {
        format!(
            "{}:{}",
            self.provider.name(),
            self.model.as_deref().unwrap_or(&request.model)
        )
    }

    fn request(&self, request: &ModelRequest) -> ModelRequest {
        let mut request = request.clone();
        if let Some(model) = &self.model {
            request.model.clone_from(model);
        }
        request
    }
}

/// Why a [`Failover`] chain moved on from a step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SwitchReason {
    /// The step failed with this error.
    Failed { error: String },
    /// No answer started within the step's deadline.
    TimedOut { after_ms: u64 },
}

/// A move from one step of a [`Failover`] chain to the next, reported as
/// [`StreamEvent::Switched`] and then as [`crate::Event::ProviderSwitched`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderSwitch {
    /// `provider:model` of the step that gave up.
    pub from: String,
    /// `provider:model` of the step tried next.
    pub to: String,
    pub reason: SwitchReason,
}

/// What a [`Failover`] does after a failed attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum FailoverAction {
    /// Try the same step again after this delay.
    Retry(Duration),
    /// Try the same step again after `delay` with `request` instead (fewer
    /// output tokens, no images, default options). Later attempts on the
    /// step keep it; the next step starts from the original request.
    RetryWith {
        delay: Duration,
        request: Box<ModelRequest>,
    },
    /// Move to the next step.
    Next,
    /// Give up: no other step would do better.
    Stop,
}

/// Decides what follows a failed attempt. The default ([`DefaultFailover`])
/// retries what [`ProviderError::is_retryable`] allows on the same step,
/// then moves on; it moves on at once from authentication, billing,
/// not-found and other errors, and stops on an invalid request, a context
/// overflow (the caller must shrink it) and a wrongly shaped reply.
/// A request with [`crate::RequestOptions::single_attempt`] stops at its
/// first error whatever the policy says.
pub trait FailoverPolicy: Send + Sync {
    /// Whether to send `request` on this step at all. An error skips the
    /// attempt and is handled as its failure (for example, a model the app
    /// knows is out of credit). Default: send it.
    fn before_attempt(&self, _request: &ModelRequest) -> Result<(), ProviderError> {
        Ok(())
    }

    /// `request` is what the failed attempt sent; `attempt` counts from 1;
    /// `attempts` is the step's limit.
    fn after_error(
        &self,
        request: &ModelRequest,
        error: &ProviderError,
        attempt: u32,
        attempts: u32,
    ) -> FailoverAction;
}

/// The default [`FailoverPolicy`].
#[derive(Debug, Clone, Copy)]
pub struct DefaultFailover {
    pub base_delay: Duration,
}

impl Default for DefaultFailover {
    fn default() -> Self {
        Self {
            base_delay: Duration::from_millis(500),
        }
    }
}

impl FailoverPolicy for DefaultFailover {
    fn after_error(
        &self,
        _request: &ModelRequest,
        error: &ProviderError,
        attempt: u32,
        attempts: u32,
    ) -> FailoverAction {
        match error {
            error if error.is_retryable() && attempt < attempts => {
                let delay = match error {
                    ProviderError::RateLimited {
                        retry_after: Some(after),
                    } => *after,
                    _ => self.base_delay * 2u32.saturating_pow(attempt.saturating_sub(1)),
                };
                FailoverAction::Retry(delay.min(MAX_BACKOFF))
            }
            ProviderError::InvalidRequest(_)
            | ProviderError::ContextOverflow { .. }
            | ProviderError::Malformed { parse: false, .. } => FailoverAction::Stop,
            _ => FailoverAction::Next,
        }
    }
}

/// A provider chain: each step is tried in order until one answers.
///
/// Unlike wrapping the chain in one timeout, every attempt gets its own
/// [`FailoverStep::deadline`], so a step that hangs cannot use up the time
/// of the steps after it; a step whose attempt hit its deadline is not
/// retried. Once a step has started streaming, the chain is
/// committed to it: switching then would duplicate visible output.
///
/// Every move to the next step is yielded as [`StreamEvent::Switched`]
/// before the next step's events; the run loop reports it as
/// [`crate::Event::ProviderSwitched`]. [`Provider::complete`] has no way
/// to report them; the run loop always streams.
///
/// Which models form the chain, and in what order, is the app's policy.
pub struct Failover {
    steps: Vec<FailoverStep>,
    policy: Arc<dyn FailoverPolicy>,
    name: String,
}

impl Failover {
    pub fn new(steps: Vec<FailoverStep>) -> Self {
        Self {
            steps,
            policy: Arc::new(DefaultFailover::default()),
            name: "failover".into(),
        }
    }

    pub fn policy(mut self, policy: impl FailoverPolicy + 'static) -> Self {
        self.policy = Arc::new(policy);
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn steps(&self) -> &[FailoverStep] {
        &self.steps
    }

    /// What follows a failed attempt that sent `sent`: retry the step
    /// (`Some`, possibly with a rewritten request), move on (`None`), or the
    /// error to give up with.
    fn next_move(
        &self,
        sent: &ModelRequest,
        step: &FailoverStep,
        attempt: u32,
        error: ProviderError,
    ) -> Result<Option<(Duration, Option<ModelRequest>)>, ProviderError> {
        if sent.options.single_attempt {
            return Err(error);
        }
        match self
            .policy
            .after_error(sent, &error, attempt, step.attempts)
        {
            FailoverAction::Retry(delay) => Ok(Some((delay, None))),
            FailoverAction::RetryWith { delay, request } => Ok(Some((delay, Some(*request)))),
            FailoverAction::Next => Ok(None),
            FailoverAction::Stop => Err(error),
        }
    }
}

fn no_steps() -> ProviderError {
    ProviderError::Other("failover chain has no steps".into())
}

fn timed_out(deadline: Duration) -> ProviderError {
    ProviderError::Transient(format!("no answer within {deadline:?}"))
}

fn reason(error: &ProviderError, deadline: Duration) -> SwitchReason {
    if *error == timed_out(deadline) {
        SwitchReason::TimedOut {
            after_ms: deadline.as_millis().min(u64::MAX as u128) as u64,
        }
    } else {
        SwitchReason::Failed {
            error: error.to_string(),
        }
    }
}

#[async_trait]
impl Provider for Failover {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(&self, request: &ModelRequest) -> Result<ModelResponse, ProviderError> {
        let mut last = None;
        for step in &self.steps {
            let mut step_request = step.request(request);
            let mut attempt = 0;
            let error = loop {
                attempt += 1;
                let error = match self.policy.before_attempt(&step_request) {
                    Err(error) => error,
                    Ok(()) => match tokio::time::timeout(
                        step.deadline,
                        step.provider.complete(&step_request),
                    )
                    .await
                    {
                        Ok(Ok(response)) => return Ok(response),
                        Ok(Err(error)) => error,
                        // A step that hung is not retried: the next step
                        // gets the time instead (a single-attempt request
                        // stops below).
                        Err(_) if !step_request.options.single_attempt => {
                            break timed_out(step.deadline)
                        }
                        Err(_) => timed_out(step.deadline),
                    },
                };
                match self.next_move(&step_request, step, attempt, error.clone())? {
                    Some((delay, rewritten)) => {
                        tokio::time::sleep(delay).await;
                        if let Some(rewritten) = rewritten {
                            step_request = rewritten;
                        }
                    }
                    None => break error,
                }
            };
            last = Some(error);
        }
        Err(last.unwrap_or_else(no_steps))
    }

    fn stream<'a>(
        &'a self,
        request: &'a ModelRequest,
    ) -> BoxStream<'a, Result<StreamEvent, ProviderError>> {
        // Each step's request is owned by the driver below and its stream
        // borrows it, so events reach the caller through a channel.
        let (tx, rx) = mpsc::unbounded::<Result<StreamEvent, ProviderError>>();
        let driver = async move {
            let mut last = None;
            for (index, step) in self.steps.iter().enumerate() {
                let mut step_request = step.request(request);
                let mut attempt = 0;
                let error = loop {
                    attempt += 1;
                    let error = match self.policy.before_attempt(&step_request) {
                        Err(error) => error,
                        Ok(()) => {
                            let mut events = step.provider.stream(&step_request);
                            match tokio::time::timeout(step.deadline, events.next()).await {
                                Ok(Some(Ok(first))) => {
                                    // Committed to this step: forward everything.
                                    if tx.unbounded_send(Ok(first)).is_err() {
                                        return;
                                    }
                                    while let Some(event) = events.next().await {
                                        if tx.unbounded_send(event).is_err() {
                                            return;
                                        }
                                    }
                                    return;
                                }
                                Ok(Some(Err(error))) => error,
                                Ok(None) => ProviderError::Other(
                                    "provider stream ended without a response".into(),
                                ),
                                // A step that hung is not retried: the next
                                // step gets the time instead (a single-attempt
                                // request stops below).
                                Err(_) if !step_request.options.single_attempt => {
                                    break timed_out(step.deadline)
                                }
                                Err(_) => timed_out(step.deadline),
                            }
                        }
                    };
                    match self.next_move(&step_request, step, attempt, error.clone()) {
                        Ok(Some((delay, rewritten))) => {
                            tokio::time::sleep(delay).await;
                            if let Some(rewritten) = rewritten {
                                step_request = rewritten;
                            }
                        }
                        Ok(None) => break error,
                        Err(error) => {
                            let _ = tx.unbounded_send(Err(error));
                            return;
                        }
                    }
                };
                if let Some(next) = self.steps.get(index + 1) {
                    let switch = ProviderSwitch {
                        from: step.label(request),
                        to: next.label(request),
                        reason: reason(&error, step.deadline),
                    };
                    if tx
                        .unbounded_send(Ok(StreamEvent::Switched(switch)))
                        .is_err()
                    {
                        return;
                    }
                }
                last = Some(error);
            }
            let _ = tx.unbounded_send(Err(last.unwrap_or_else(no_steps)));
        };
        // The driver ends after its last send, so the merged stream ends
        // once the channel drains.
        Box::pin(stream::select(
            rx,
            stream::once(driver).filter_map(|()| async { None }),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{text, ScriptedProvider};

    /// Never answers.
    struct Hanging;

    #[async_trait]
    impl Provider for Hanging {
        fn name(&self) -> &str {
            "hanging"
        }

        async fn complete(&self, _: &ModelRequest) -> Result<ModelResponse, ProviderError> {
            std::future::pending().await
        }
    }

    fn request() -> ModelRequest {
        ModelRequest {
            model: "primary-model".into(),
            messages: vec![crate::Message::user("hello")],
            tools: Vec::new(),
            tool_choice: Default::default(),
            max_output_tokens: None,
            ..Default::default()
        }
    }

    async fn collect(provider: &Failover) -> Vec<Result<StreamEvent, ProviderError>> {
        let request = request();
        provider.stream(&request).collect().await
    }

    /// The primary hangs; the fallback answers within its own deadline, and
    /// the switch is reported before the answer.
    #[tokio::test(start_paused = true)]
    async fn a_hanging_primary_falls_back_within_the_fallbacks_own_deadline() {
        let fallback = ScriptedProvider::new(vec![text("from the fallback")]);
        let chain = Failover::new(vec![
            FailoverStep::new(Hanging).deadline(Duration::from_secs(30)),
            FailoverStep::new(fallback)
                .model("fallback-model")
                .deadline(Duration::from_secs(30)),
        ]);
        let started = tokio::time::Instant::now();
        let events = collect(&chain).await;
        assert_eq!(started.elapsed(), Duration::from_secs(30));
        assert_eq!(
            events[0],
            Ok(StreamEvent::Switched(ProviderSwitch {
                from: "hanging:primary-model".into(),
                to: "scripted:fallback-model".into(),
                reason: SwitchReason::TimedOut { after_ms: 30_000 },
            }))
        );
        match events.last() {
            Some(Ok(StreamEvent::Done(response))) => assert_eq!(response.text, "from the fallback"),
            other => panic!("expected the fallback's answer, got {other:?}"),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn complete_gives_each_attempt_its_own_deadline() {
        let chain = Failover::new(vec![
            FailoverStep::new(Hanging).deadline(Duration::from_secs(5)),
            FailoverStep::new(ScriptedProvider::new(vec![text("ok")])),
        ]);
        let response = chain.complete(&request()).await.unwrap();
        assert_eq!(response.text, "ok");
    }

    #[tokio::test(start_paused = true)]
    async fn transient_errors_retry_the_same_step_before_moving_on() {
        let flaky = ScriptedProvider::with_results(vec![
            Err(ProviderError::Transient("blip".into())),
            Ok(text("second try")),
        ]);
        let chain = Failover::new(vec![
            FailoverStep::new(flaky).attempts(2),
            FailoverStep::new(ScriptedProvider::new(vec![text("fallback")])),
        ]);
        let events = collect(&chain).await;
        assert!(!events
            .iter()
            .any(|e| matches!(e, Ok(StreamEvent::Switched(_)))));
        assert!(matches!(events.last(), Some(Ok(StreamEvent::Done(r))) if r.text == "second try"));
    }

    #[tokio::test]
    async fn an_invalid_request_stops_the_chain() {
        let chain = Failover::new(vec![
            FailoverStep::new(ScriptedProvider::with_results(vec![Err(
                ProviderError::InvalidRequest("bad schema".into()),
            )])),
            FailoverStep::new(ScriptedProvider::new(vec![text("never")])),
        ]);
        let events = collect(&chain).await;
        assert_eq!(
            events,
            vec![Err(ProviderError::InvalidRequest("bad schema".into()))]
        );
    }

    #[tokio::test]
    async fn when_every_step_fails_the_last_error_ends_the_stream() {
        let chain = Failover::new(vec![
            FailoverStep::new(ScriptedProvider::with_results(vec![Err(
                ProviderError::Auth("expired".into()),
            )])),
            FailoverStep::new(ScriptedProvider::with_results(vec![Err(
                ProviderError::Other("down".into()),
            )])),
        ]);
        let events = collect(&chain).await;
        assert!(
            matches!(&events[0], Ok(StreamEvent::Switched(s)) if s.reason == SwitchReason::Failed { error: "authentication failed: expired".into() })
        );
        assert_eq!(
            events.last(),
            Some(&Err(ProviderError::Other("down".into())))
        );
        assert_eq!(events.len(), 2);
    }

    /// Through an agent run, the switch is a typed run event.
    #[tokio::test(start_paused = true)]
    async fn a_run_reports_the_switch_as_an_event() {
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = {
            let events = events.clone();
            move |_: crate::RunId, event: &crate::Event| events.lock().unwrap().push(event.clone())
        };
        let agent = crate::Agent::builder("failover")
            .model("primary-model")
            .provider(Failover::new(vec![
                FailoverStep::new(Hanging).deadline(Duration::from_secs(10)),
                FailoverStep::new(ScriptedProvider::new(vec![text("answered")])),
            ]))
            .event_sink(sink)
            .build()
            .unwrap();
        let outcome = agent.run("cli:synthetic-user-1", "hello").await.unwrap();
        assert_eq!(outcome.final_text.as_deref(), Some("answered"));
        let events = events.lock().unwrap();
        assert!(events.iter().any(|event| matches!(
            event,
            crate::Event::ProviderSwitched { turn: 1, switch }
                if switch.from == "hanging:primary-model"
                    && switch.reason == SwitchReason::TimedOut { after_ms: 10_000 }
        )));
    }

    /// Retries once with an output cap after a billing error, and skips a
    /// model it knows is out of credit without calling it.
    struct CapThenSkip;

    impl FailoverPolicy for CapThenSkip {
        fn before_attempt(&self, request: &ModelRequest) -> Result<(), ProviderError> {
            if request.model == "broke-model" {
                return Err(ProviderError::Billing {
                    message: "cached".into(),
                    affordable_tokens: None,
                });
            }
            Ok(())
        }

        fn after_error(
            &self,
            request: &ModelRequest,
            error: &ProviderError,
            _attempt: u32,
            _attempts: u32,
        ) -> FailoverAction {
            match error {
                ProviderError::Billing {
                    affordable_tokens: Some(tokens),
                    ..
                } if request.max_output_tokens.is_none() => {
                    let mut capped = request.clone();
                    capped.max_output_tokens = Some(*tokens as u32);
                    FailoverAction::RetryWith {
                        delay: Duration::ZERO,
                        request: Box::new(capped),
                    }
                }
                _ => FailoverAction::Next,
            }
        }
    }

    #[tokio::test]
    async fn a_policy_can_retry_with_a_rewritten_request() {
        let primary = Arc::new(ScriptedProvider::with_results(vec![
            Err(ProviderError::Billing {
                message: "402".into(),
                affordable_tokens: Some(600),
            }),
            Ok(text("capped answer")),
        ]));
        let chain = Failover::new(vec![FailoverStep::from_arc(primary.clone()).attempts(3)])
            .policy(CapThenSkip);
        let events = collect(&chain).await;
        assert!(
            matches!(events.last(), Some(Ok(StreamEvent::Done(r))) if r.text == "capped answer")
        );
        let sent = primary.requests();
        assert_eq!(sent[0].max_output_tokens, None);
        assert_eq!(sent[1].max_output_tokens, Some(600));
    }

    #[tokio::test]
    async fn a_policy_can_skip_an_attempt_without_sending_it() {
        let broke = Arc::new(ScriptedProvider::new(vec![text("never")]));
        let chain = Failover::new(vec![
            FailoverStep::from_arc(broke.clone()).model("broke-model"),
            FailoverStep::new(ScriptedProvider::new(vec![text("from the fallback")])),
        ])
        .policy(CapThenSkip);
        let events = collect(&chain).await;
        assert!(broke.requests().is_empty());
        assert!(
            matches!(&events[0], Ok(StreamEvent::Switched(s)) if s.from == "scripted:broke-model")
        );
        assert!(
            matches!(events.last(), Some(Ok(StreamEvent::Done(r))) if r.text == "from the fallback")
        );
    }

    /// A hung step is not retried, even when its policy would retry a
    /// transient error; a single-attempt request stops at the first one.
    #[tokio::test(start_paused = true)]
    async fn a_timed_out_step_moves_on_unless_the_request_is_single_attempt() {
        let chain = || {
            Failover::new(vec![
                FailoverStep::new(Hanging)
                    .deadline(Duration::from_secs(5))
                    .attempts(5),
                FailoverStep::new(ScriptedProvider::new(vec![text("fallback")])),
            ])
        };
        let started = tokio::time::Instant::now();
        let events = collect(&chain()).await;
        assert_eq!(started.elapsed(), Duration::from_secs(5));
        assert!(matches!(events.last(), Some(Ok(StreamEvent::Done(r))) if r.text == "fallback"));

        let mut single = request();
        single.options.single_attempt = true;
        let events: Vec<_> = chain().stream(&single).collect().await;
        assert!(matches!(
            events.as_slice(),
            [Err(ProviderError::Transient(_))]
        ));
    }

    #[test]
    fn the_default_policy_caps_backoff_and_honours_retry_after() {
        let policy = DefaultFailover::default();
        let limited = ProviderError::RateLimited {
            retry_after: Some(Duration::from_secs(120)),
        };
        assert_eq!(
            policy.after_error(&request(), &limited, 1, 3),
            FailoverAction::Retry(MAX_BACKOFF)
        );
        assert_eq!(
            policy.after_error(&request(), &limited, 3, 3),
            FailoverAction::Next
        );
        let blip = ProviderError::Transient("x".into());
        assert_eq!(
            policy.after_error(&request(), &blip, 2, 3),
            FailoverAction::Retry(Duration::from_secs(1))
        );
    }
}
