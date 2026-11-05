//! Conversation compaction across runs: older whole turns are replaced by a
//! rolling summary once the transcript nears the model's context window.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_core::{Hook, HookContext, Message, ModelRequest, Provider, Role, RunOutcome};
use async_trait::async_trait;
use tokio::task::JoinHandle;

use crate::context_window::estimate_tokens;

/// The rolling summary of a session's transcript up to `covered` messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationSummary {
    pub text: String,
    /// Leading transcript messages the summary stands for. Always a turn
    /// boundary: the message at `covered` is a user message.
    pub covered: usize,
    /// Fingerprint of `transcript[..covered]`. A transcript that no longer
    /// starts with those messages (a new conversation in the same session,
    /// an edited history) does not use the summary.
    pub fingerprint: u64,
}

/// Where summaries live. Implement it over your own database to keep them
/// across restarts; [`InMemorySummaries`] keeps them for the process.
#[async_trait]
pub trait SummaryStore: Send + Sync {
    async fn load(&self, session_id: &str) -> Option<ConversationSummary>;
    async fn save(&self, session_id: &str, summary: ConversationSummary);
}

/// Process-local [`SummaryStore`].
#[derive(Default)]
pub struct InMemorySummaries {
    summaries: Mutex<HashMap<String, ConversationSummary>>,
}

#[async_trait]
impl SummaryStore for InMemorySummaries {
    async fn load(&self, session_id: &str) -> Option<ConversationSummary> {
        self.summaries.lock().ok()?.get(session_id).cloned()
    }

    async fn save(&self, session_id: &str, summary: ConversationSummary) {
        if let Ok(mut summaries) = self.summaries.lock() {
            summaries.insert(session_id.to_string(), summary);
        }
    }
}

/// Writes the summary: folds `messages` into `previous` (the summary of
/// everything before them) and returns the new cumulative summary.
#[async_trait]
pub trait Summarizer: Send + Sync {
    async fn summarize(
        &self,
        previous: Option<&str>,
        messages: &[Message],
    ) -> Result<String, String>;
}

const SUMMARY_INSTRUCTIONS: &str = "You maintain the running state of a conversation \
between a user and an AI agent so the agent can continue it without the full transcript. \
Merge the existing state (if any) with the new messages and return only the updated state, \
in plain text under these headings: Goals and open loops; Decisions and constraints the user \
set; Facts, names, paths and identifiers; Work done and results; Pending questions. Keep \
every concrete detail the agent may need later (file paths, IDs, numbers, commands, \
errors). Drop pleasantries and repetition. Never invent anything not in the input.";

/// Characters kept from each tool result when it is handed to the
/// summarizer; the rest is marked elided.
const MAX_TOOL_RESULT_CHARS: usize = 2_000;

/// A [`Summarizer`] backed by any [`Provider`], usually a fast, cheap model.
pub struct ProviderSummarizer {
    provider: Arc<dyn Provider>,
    model: String,
    max_output_tokens: u32,
}

impl ProviderSummarizer {
    pub fn new(provider: Arc<dyn Provider>, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            max_output_tokens: 1_500,
        }
    }

    pub fn max_output_tokens(mut self, tokens: u32) -> Self {
        self.max_output_tokens = tokens;
        self
    }
}

#[async_trait]
impl Summarizer for ProviderSummarizer {
    async fn summarize(
        &self,
        previous: Option<&str>,
        messages: &[Message],
    ) -> Result<String, String> {
        let mut input = String::new();
        if let Some(previous) = previous {
            input.push_str("Existing state:\n");
            input.push_str(previous);
            input.push_str("\n\n");
        }
        input.push_str("New messages:\n");
        for message in messages {
            input.push_str(&render(message));
            input.push('\n');
        }
        let request = ModelRequest {
            model: self.model.clone(),
            messages: vec![Message::system(SUMMARY_INSTRUCTIONS), Message::user(input)],
            tools: Vec::new(),
            tool_choice: Default::default(),
            max_output_tokens: Some(self.max_output_tokens),
        };
        let response = self
            .provider
            .complete(&request)
            .await
            .map_err(|e| e.to_string())?;
        tracing::info!(
            model = %self.model,
            input_tokens = response.usage.input_tokens,
            output_tokens = response.usage.output_tokens,
            "conversation summary written"
        );
        let text = response.text.trim().to_string();
        if text.is_empty() {
            return Err("the summarizer returned an empty summary".into());
        }
        Ok(text)
    }
}

fn render(message: &Message) -> String {
    let role = match message.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool result",
    };
    let mut content = message.content.clone();
    if message.role == Role::Tool && content.chars().count() > MAX_TOOL_RESULT_CHARS {
        let total = content.chars().count();
        content = content.chars().take(MAX_TOOL_RESULT_CHARS).collect();
        content.push_str(&format!(
            " [... {} more chars elided]",
            total - MAX_TOOL_RESULT_CHARS
        ));
    }
    let mut out = format!("[{role}] {content}");
    for call in &message.tool_calls {
        out.push_str(&format!("\n[tool call] {} {}", call.name, call.arguments));
    }
    out
}

/// Keeps a session's transcript inside the model's context window by
/// summarizing older whole turns, the way a long chat should age.
///
/// - **After each run** (`on_run_end`), when the unsummarized transcript
///   passes 60% of the window, older turns are folded into the session's
///   summary on a background task. Every message is final once a run has
///   ended, so the next run normally finds the work done.
/// - **Before each model call** (`before_model`), the summarized prefix is
///   replaced, in the request only, by one system note with the summary.
///   The stored transcript is never changed. Only when the request is still
///   past 80% of the window does the call wait: for a background pass still
///   running, else for an inline one (bounded by [`Self::wait_timeout`]).
///
/// Summaries cut only before a user message, so a tool call is never
/// separated from its result, and about a quarter of the threshold of recent
/// turns always stays verbatim. The window is per model: the request's model
/// name is matched against [`Self::model_window`] entries, else the default.
///
/// Register it before [`crate::ContextWindow`], which then trims whatever
/// still does not fit.
pub struct ConversationCompaction {
    inner: Arc<Inner>,
}

struct Inner {
    summarizer: Arc<dyn Summarizer>,
    store: Arc<dyn SummaryStore>,
    default_window: usize,
    model_windows: Vec<(String, usize)>,
    wait_timeout: Duration,
    /// The model each session last called, for the window after a run.
    session_models: Mutex<HashMap<String, String>>,
    /// One compaction at a time per session.
    session_locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    jobs: Mutex<HashMap<String, JoinHandle<()>>>,
}

impl ConversationCompaction {
    /// `default_window` is the context window, in tokens, of models without
    /// a [`Self::model_window`] entry.
    pub fn new(
        summarizer: impl Summarizer + 'static,
        store: impl SummaryStore + 'static,
        default_window: usize,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                summarizer: Arc::new(summarizer),
                store: Arc::new(store),
                default_window,
                model_windows: Vec::new(),
                wait_timeout: Duration::from_secs(30),
                session_models: Mutex::new(HashMap::new()),
                session_locks: Mutex::new(HashMap::new()),
                jobs: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// The window of models whose name starts with `prefix` (ignoring case
    /// and a provider route such as `openrouter/`); the longest prefix wins.
    pub fn model_window(mut self, prefix: impl Into<String>, tokens: usize) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("configured before use")
            .model_windows
            .push((prefix.into().to_ascii_lowercase(), tokens));
        self
    }

    /// The longest a model call waits for compaction (default 30s). After
    /// it, the call goes ahead and [`crate::ContextWindow`] trims.
    pub fn wait_timeout(mut self, timeout: Duration) -> Self {
        Arc::get_mut(&mut self.inner)
            .expect("configured before use")
            .wait_timeout = timeout;
        self
    }

    /// The context window used for `model`.
    pub fn window_for(&self, model: &str) -> usize {
        self.inner.window_for(model)
    }

    /// The session's current summary.
    pub async fn summary(&self, session_id: &str) -> Option<ConversationSummary> {
        self.inner.store.load(session_id).await
    }

    /// Wait for every background pass started so far.
    pub async fn drain(&self) {
        let jobs: Vec<JoinHandle<()>> = self
            .inner
            .jobs
            .lock()
            .map(|mut jobs| jobs.drain().map(|(_, job)| job).collect())
            .unwrap_or_default();
        for job in jobs {
            let _ = job.await;
        }
    }
}

impl Inner {
    fn window_for(&self, model: &str) -> usize {
        let model = model.to_ascii_lowercase();
        let name = model.rsplit('/').next().unwrap_or(&model);
        self.model_windows
            .iter()
            .filter(|(prefix, _)| name.starts_with(prefix.as_str()))
            .max_by_key(|(prefix, _)| prefix.len())
            .map_or(self.default_window, |(_, tokens)| *tokens)
    }

    fn compact_at(&self, model: &str) -> usize {
        self.window_for(model).saturating_mul(3) / 5
    }

    fn wait_at(&self, model: &str) -> usize {
        self.window_for(model).saturating_mul(4) / 5
    }

    fn session_lock(&self, session_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self
            .session_locks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        locks.entry(session_id.to_string()).or_default().clone()
    }

    /// The stored summary, if `transcript` still starts with what it covers.
    async fn valid_summary(
        &self,
        session_id: &str,
        transcript: &[Message],
    ) -> Option<ConversationSummary> {
        let summary = self.store.load(session_id).await?;
        (summary.covered <= transcript.len()
            && fingerprint(&transcript[..summary.covered]) == summary.fingerprint)
            .then_some(summary)
    }

    /// Fold older turns into the summary when the unsummarized transcript
    /// passes the threshold for `model`. Returns whether it compacted.
    async fn compact(&self, session_id: &str, transcript: &[Message], model: &str) -> bool {
        let lock = self.session_lock(session_id);
        let _guard = lock.lock().await;
        let summary = self.valid_summary(session_id, transcript).await;
        let covered = summary.as_ref().map_or(0, |s| s.covered);
        let summary_tokens = summary
            .as_ref()
            .map_or(0, |s| s.text.chars().count().div_ceil(4));
        let threshold = self.compact_at(model);
        if estimate_tokens(&transcript[covered..]) + summary_tokens < threshold {
            return false;
        }
        let Some(cut) = cut_point(transcript, covered, threshold / 4) else {
            return false;
        };
        let previous = summary.as_ref().map(|s| s.text.as_str());
        match self
            .summarizer
            .summarize(previous, &transcript[covered..cut])
            .await
        {
            Ok(text) => {
                self.store
                    .save(
                        session_id,
                        ConversationSummary {
                            text,
                            covered: cut,
                            fingerprint: fingerprint(&transcript[..cut]),
                        },
                    )
                    .await;
                true
            }
            Err(error) => {
                tracing::warn!(session_id, %error, "conversation compaction failed");
                false
            }
        }
    }

    /// Replace the summarized prefix of the transcript inside `request`.
    /// Returns false when the request does not carry the transcript as
    /// expected (another hook reshaped it), leaving it unchanged.
    fn apply(
        &self,
        request: &mut ModelRequest,
        transcript: &[Message],
        summary: &ConversationSummary,
    ) -> bool {
        if summary.covered == 0 || transcript.is_empty() {
            return false;
        }
        let Some(start) = request.messages.iter().position(|m| *m == transcript[0]) else {
            return false;
        };
        let end = start + summary.covered;
        if request.messages.len() < end
            || request.messages[start..end] != transcript[..summary.covered]
        {
            return false;
        }
        let note = Message::system(format!(
            "[Earlier in this conversation ({} messages), summarized. The messages after this \
             note are verbatim.]\n{}",
            summary.covered, summary.text
        ));
        request.messages.splice(start..end, [note]);
        true
    }
}

/// The turn boundary to summarize up to: the earliest user message after
/// `covered` whose turn and everything after it fit in `keep_tokens`. The
/// newest turn is always kept, so `None` when there is nothing older.
fn cut_point(transcript: &[Message], covered: usize, keep_tokens: usize) -> Option<usize> {
    let boundaries: Vec<usize> = transcript
        .iter()
        .enumerate()
        .skip(covered + 1)
        .filter(|(_, m)| m.role == Role::User)
        .map(|(i, _)| i)
        .collect();
    let last = *boundaries.last()?;
    Some(
        boundaries
            .into_iter()
            .find(|&i| estimate_tokens(&transcript[i..]) <= keep_tokens)
            .unwrap_or(last),
    )
}

fn fingerprint(messages: &[Message]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for message in messages {
        serde_json::to_string(message)
            .unwrap_or_default()
            .hash(&mut hasher);
    }
    hasher.finish()
}

#[async_trait]
impl Hook for ConversationCompaction {
    async fn before_model(&self, ctx: &HookContext<'_>, request: &mut ModelRequest) {
        let inner = &self.inner;
        if let Ok(mut models) = inner.session_models.lock() {
            models.insert(ctx.session_id.to_string(), request.model.clone());
        }
        let transcript = ctx.messages;
        let wait_at = inner.wait_at(&request.model);
        let unsummarized = |summary: &Option<ConversationSummary>| {
            let covered = summary.as_ref().map_or(0, |s| s.covered);
            estimate_tokens(&transcript[covered..])
                + summary
                    .as_ref()
                    .map_or(0, |s| s.text.chars().count().div_ceil(4))
        };
        let mut summary = inner.valid_summary(ctx.session_id, transcript).await;
        if unsummarized(&summary) > wait_at {
            // Near the window: let a running pass finish, else compact now.
            let job = inner
                .jobs
                .lock()
                .ok()
                .and_then(|mut jobs| jobs.remove(ctx.session_id));
            let wait = async {
                match job {
                    Some(job) => {
                        let _ = job.await;
                    }
                    None => {
                        inner
                            .compact(ctx.session_id, transcript, &request.model)
                            .await;
                    }
                }
            };
            if tokio::time::timeout(inner.wait_timeout, wait)
                .await
                .is_err()
            {
                tracing::warn!(
                    session_id = ctx.session_id,
                    "conversation compaction did not finish in time; sending the request as is"
                );
            }
            summary = inner.valid_summary(ctx.session_id, transcript).await;
        }
        if let Some(summary) = summary {
            inner.apply(request, transcript, &summary);
        }
    }

    async fn on_run_end(&self, ctx: &HookContext<'_>, _outcome: &RunOutcome) {
        let inner = self.inner.clone();
        let session_id = ctx.session_id.to_string();
        let model = inner
            .session_models
            .lock()
            .ok()
            .and_then(|models| models.get(&session_id).cloned())
            .unwrap_or_default();
        let transcript = ctx.messages.to_vec();
        let job = tokio::spawn({
            let inner = inner.clone();
            let session_id = session_id.clone();
            async move {
                inner.compact(&session_id, &transcript, &model).await;
            }
        });
        // A newer pass supersedes the handle; the older one still runs and
        // the per-session lock orders them.
        let mut jobs = inner
            .jobs
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        jobs.retain(|_, job| !job.is_finished());
        jobs.insert(session_id, job);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aidaemon_core::ToolCall;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The previous summary and the messages of each call.
    type Calls = Arc<Mutex<Vec<(Option<String>, Vec<Message>)>>>;

    /// Records what it was asked to fold in.
    #[derive(Default, Clone)]
    struct Recording {
        calls: Calls,
        delay: Option<Duration>,
    }

    #[async_trait]
    impl Summarizer for Recording {
        async fn summarize(
            &self,
            previous: Option<&str>,
            messages: &[Message],
        ) -> Result<String, String> {
            if let Some(delay) = self.delay {
                tokio::time::sleep(delay).await;
            }
            let mut calls = self.calls.lock().unwrap();
            calls.push((previous.map(str::to_string), messages.to_vec()));
            Ok(format!("summary #{}", calls.len()))
        }
    }

    /// A turn of roughly `tokens` tokens: user message, a tool round, reply.
    fn turn(n: usize, tokens: usize) -> Vec<Message> {
        let id = format!("call-{n}");
        vec![
            Message::user(format!("Question {n} from Alice Rivera")),
            Message::assistant(
                "",
                vec![ToolCall::new(&id, "read_file", json!({"path": "notes.md"}))],
            ),
            Message::tool_result(&id, "x".repeat(tokens * 4)),
            Message::assistant(format!("Answer {n}"), vec![]),
        ]
    }

    fn transcript(turns: usize, tokens: usize) -> Vec<Message> {
        (0..turns).flat_map(|n| turn(n, tokens)).collect()
    }

    fn request(model: &str, transcript: &[Message]) -> ModelRequest {
        let mut messages = vec![Message::system("You are a helpful agent.")];
        messages.extend(transcript.iter().cloned());
        ModelRequest {
            model: model.into(),
            messages,
            tools: Vec::new(),
            tool_choice: Default::default(),
            max_output_tokens: None,
        }
    }

    fn compaction(summarizer: Recording, window: usize) -> ConversationCompaction {
        ConversationCompaction::new(summarizer, InMemorySummaries::default(), window)
    }

    #[test]
    fn window_follows_the_longest_matching_model_prefix() {
        let c = compaction(Recording::default(), 128_000)
            .model_window("gpt-6", 272_000)
            .model_window("gpt-6-mini", 64_000)
            .model_window("claude-", 200_000);
        assert_eq!(c.window_for("gpt-6-sol"), 272_000);
        assert_eq!(c.window_for("GPT-6-mini-2026"), 64_000);
        assert_eq!(c.window_for("openrouter/claude-sonnet-5-5"), 200_000);
        assert_eq!(c.window_for("local-model"), 128_000);
    }

    #[test]
    fn cuts_only_at_user_messages_and_keeps_the_newest_turn() {
        let t = transcript(4, 1_000);
        // Keep about two turns verbatim.
        let cut = cut_point(&t, 0, 2_100).unwrap();
        assert_eq!(t[cut].role, Role::User);
        assert_eq!(cut, 8);
        // Nothing fits: still keep the newest turn.
        assert_eq!(cut_point(&t, 0, 0), Some(12));
        // A single turn has nothing older to summarize.
        assert_eq!(cut_point(&turn(0, 1_000), 0, 0), None);
    }

    /// Below 60% of the window nothing is summarized, whatever the turn
    /// count: compaction is driven by the window, not by every turn.
    #[tokio::test]
    async fn below_the_threshold_nothing_is_summarized() {
        let summarizer = Recording::default();
        let c = compaction(summarizer.clone(), 100_000);
        assert!(!c.inner.compact("s", &transcript(20, 2_500), "m").await);
        assert!(summarizer.calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn the_same_history_compacts_for_a_small_window_only() {
        let t = transcript(10, 1_000);
        let large = compaction(Recording::default(), 200_000);
        assert!(!large.inner.compact("s", &t, "big").await);
        let small = compaction(Recording::default(), 200_000).model_window("small", 12_000);
        assert!(small.inner.compact("s", &t, "small-model").await);
    }

    /// The pass after a run leaves the summary ready, so the next run's
    /// model call substitutes it without waiting or summarizing.
    #[tokio::test]
    async fn the_next_run_finds_the_summary_ready() {
        let summarizer = Recording::default();
        let c = compaction(summarizer.clone(), 10_000);
        let t = transcript(8, 1_000);
        assert!(c.inner.compact("s", &t, "m").await);
        let summary = c.summary("s").await.unwrap();
        assert_eq!(t[summary.covered].role, Role::User);
        assert!(summary.covered < t.len());

        let mut next = t.clone();
        next.push(Message::user("Next question"));
        let mut req = request("m", &next);
        let calls_before = summarizer.calls.lock().unwrap().len();
        assert!(c.inner.apply(&mut req, &next, &summary));
        assert_eq!(summarizer.calls.lock().unwrap().len(), calls_before);
        assert_eq!(req.messages[0].content, "You are a helpful agent.");
        assert_eq!(req.messages[1].role, Role::System);
        assert!(req.messages[1].content.contains("summary #1"));
        assert_eq!(req.messages[2], next[summary.covered]);
        assert_eq!(req.messages.len(), 2 + next.len() - summary.covered);
    }

    #[tokio::test]
    async fn a_second_pass_folds_into_the_previous_summary() {
        let summarizer = Recording::default();
        let c = compaction(summarizer.clone(), 10_000);
        let mut t = transcript(8, 1_000);
        assert!(c.inner.compact("s", &t, "m").await);
        let first = c.summary("s").await.unwrap();
        t.extend(transcript(8, 1_000));
        assert!(c.inner.compact("s", &t, "m").await);
        let second = c.summary("s").await.unwrap();
        assert!(second.covered > first.covered);
        let calls = summarizer.calls.lock().unwrap();
        assert_eq!(calls[1].0.as_deref(), Some("summary #1"));
        // Only the messages after the first summary are sent again.
        assert_eq!(calls[1].1[0], t[first.covered]);
    }

    #[tokio::test]
    async fn a_different_history_does_not_use_the_summary() {
        let c = compaction(Recording::default(), 10_000);
        let t = transcript(8, 1_000);
        assert!(c.inner.compact("s", &t, "m").await);
        let mut other = transcript(8, 1_000);
        other[0] = Message::user("A different conversation");
        assert!(c.inner.valid_summary("s", &other).await.is_none());
    }

    /// Through the hook, as an agent uses it: the run's end compacts in the
    /// background and the next request carries the summary.
    #[tokio::test]
    async fn run_end_compacts_and_next_request_uses_it() {
        use aidaemon_core::testing::{text, ScriptedProvider};
        use aidaemon_core::Agent;

        let summarizer = Recording::default();
        let compaction = Arc::new(compaction(summarizer.clone(), 3_000));
        let seen = Arc::new(Mutex::new(Vec::<ModelRequest>::new()));
        struct Capture(Arc<Mutex<Vec<ModelRequest>>>);
        #[async_trait]
        impl Hook for Capture {
            async fn before_model(&self, _ctx: &HookContext<'_>, request: &mut ModelRequest) {
                self.0.lock().unwrap().push(request.clone());
            }
        }
        let long_answer = "detail ".repeat(400);
        let agent = Agent::builder("compacting")
            .model("synthetic-model")
            .provider(ScriptedProvider::new(vec![
                text(&long_answer),
                text(&long_answer),
                text(&long_answer),
                text("short"),
            ]))
            .hook_arc(compaction.clone())
            .hook(Capture(seen.clone()))
            .build()
            .unwrap();

        let mut outcome = agent
            .run("telegram:synthetic-user-1", "first")
            .await
            .unwrap();
        for question in ["second", "third", "fourth"] {
            compaction.drain().await;
            outcome = agent
                .run_from(
                    outcome.state.with_user_message(question),
                    Default::default(),
                )
                .await
                .unwrap();
        }
        compaction.drain().await;
        assert!(!summarizer.calls.lock().unwrap().is_empty());
        let last = seen.lock().unwrap().last().cloned().unwrap();
        assert!(last
            .messages
            .iter()
            .any(|m| m.role == Role::System && m.content.contains("summarized")));
        assert_eq!(last.messages.last().unwrap().content, "fourth");
    }

    fn ctx<'a>(session_id: &'a str, messages: &'a [Message]) -> HookContext<'a> {
        HookContext {
            run_id: aidaemon_core::RunId::new(),
            agent_id: "agent",
            session_id,
            turn: 1,
            messages,
            receipts: &[],
            usage: Default::default(),
            run_usage: Default::default(),
            untrusted_seen: false,
        }
    }

    #[derive(Clone)]
    struct Counting(Arc<AtomicUsize>, Duration);

    #[async_trait]
    impl Summarizer for Counting {
        async fn summarize(&self, _: Option<&str>, _: &[Message]) -> Result<String, String> {
            tokio::time::sleep(self.1).await;
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok("folded".into())
        }
    }

    /// Between 60% and 80% a model call never waits: it uses whatever
    /// summary exists and leaves compaction to the pass after the run.
    #[tokio::test]
    async fn below_the_wait_threshold_a_call_never_summarizes() {
        let count = Arc::new(AtomicUsize::new(0));
        let c = ConversationCompaction::new(
            Counting(count.clone(), Duration::ZERO),
            InMemorySummaries::default(),
            10_000,
        );
        let t = transcript(7, 1_000); // about 70% of the window
        let mut req = request("m", &t);
        c.before_model(&ctx("s", &t), &mut req).await;
        assert_eq!(count.load(Ordering::SeqCst), 0);
        assert_eq!(req.messages.len(), 1 + t.len());
    }

    /// Past 80% with a background pass still running, the model call waits
    /// for it instead of starting a second summary, then uses its result.
    #[tokio::test]
    async fn near_the_window_a_call_waits_for_the_running_pass() {
        let count = Arc::new(AtomicUsize::new(0));
        let c = ConversationCompaction::new(
            Counting(count.clone(), Duration::from_millis(50)),
            InMemorySummaries::default(),
            10_000,
        );
        let t = transcript(10, 1_000);
        c.on_run_end(&ctx("s", &t), &dummy_outcome()).await;
        let mut req = request("m", &t);
        c.before_model(&ctx("s", &t), &mut req).await;
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert!(req.messages[1].content.contains("folded"));
        assert!(req.messages.len() < 1 + t.len());
    }

    /// Past 80% with nothing running (the background pass failed or the
    /// session moved to a smaller-window model), the call compacts inline.
    #[tokio::test]
    async fn near_the_window_with_nothing_running_the_call_compacts() {
        let count = Arc::new(AtomicUsize::new(0));
        let c = ConversationCompaction::new(
            Counting(count.clone(), Duration::ZERO),
            InMemorySummaries::default(),
            200_000,
        )
        .model_window("small", 10_000);
        let t = transcript(10, 1_000);
        let mut req = request("small-model", &t);
        c.before_model(&ctx("s", &t), &mut req).await;
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert!(req.messages[1].content.contains("folded"));
    }

    /// A summarizer slower than the wait timeout does not hold the call.
    #[tokio::test]
    async fn a_slow_summary_does_not_hold_the_call_past_the_timeout() {
        let c = ConversationCompaction::new(
            Counting(Arc::default(), Duration::from_secs(60)),
            InMemorySummaries::default(),
            10_000,
        )
        .wait_timeout(Duration::from_millis(20));
        let t = transcript(10, 1_000);
        let mut req = request("m", &t);
        let started = std::time::Instant::now();
        c.before_model(&ctx("s", &t), &mut req).await;
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(req.messages.len(), 1 + t.len());
    }

    fn dummy_outcome() -> RunOutcome {
        RunOutcome {
            status: aidaemon_core::RunStatus::Completed,
            final_text: None,
            state: aidaemon_core::RunState::new("agent", "s", "hi"),
        }
    }
}
