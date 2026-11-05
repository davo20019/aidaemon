//! The hooks that put compaction into an agent: [`ConversationCompaction`]
//! for a transcript that carries the whole conversation, [`SummaryNote`] for
//! an app that already leaves summarized turns out of the transcript.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aidaemon_core::{Hook, HookContext, Message, ModelRequest, Role, RunId, RunOutcome};
use async_trait::async_trait;

use super::{
    wait_threshold, CompactionLimits, Compactor, ConversationSource, ConversationSummary,
    ConversationTurn, PendingTurns, Summarizer, SummaryStore,
};
use crate::context_window::estimate_tokens;
use crate::insert_after_system;

/// The system note that stands for the summarized turns. `guidance` is
/// appended to the coverage line (for example, which tool retrieves the
/// exact older wording).
pub fn render_summary_note(summary: &ConversationSummary, guidance: Option<&str>) -> String {
    let mut note = format!(
        "[Context Coverage]\nCompacted through turn {} / message {}. Recent raw turns and the \
         immediately preceding exchange are supplied separately.",
        summary.cursor,
        summary.last_message_id.as_deref().unwrap_or("-"),
    );
    if let Some(guidance) = guidance {
        note.push(' ');
        note.push_str(guidance);
    }
    note.push_str("\n\n[Compacted Conversation State]\n");
    note.push_str(&summary.text);
    note
}

/// Places a session's summary in each model request, after the leading
/// system messages, for an app that builds its transcript without the
/// summarized turns itself. The summary is read once per run, so every call
/// of a run sends the same note and the provider's prompt cache holds.
pub struct SummaryNote {
    summary: Source,
    guidance: Option<String>,
    pinned: Mutex<HashMap<RunId, Option<ConversationSummary>>>,
}

enum Source {
    Store(Arc<dyn SummaryStore>),
    Fixed(Option<ConversationSummary>),
}

impl SummaryNote {
    /// The session's summary from `store`, as it is when a run starts.
    pub fn new(store: Arc<dyn SummaryStore>) -> Self {
        Self {
            summary: Source::Store(store),
            guidance: None,
            pinned: Mutex::new(HashMap::new()),
        }
    }

    /// Exactly this summary, for an app that loaded it when it chose which
    /// turns to leave out, so the note and the transcript agree.
    pub fn fixed(summary: Option<ConversationSummary>) -> Self {
        Self {
            summary: Source::Fixed(summary),
            guidance: None,
            pinned: Mutex::new(HashMap::new()),
        }
    }

    /// Appended to the note's coverage line.
    pub fn guidance(mut self, guidance: impl Into<String>) -> Self {
        self.guidance = Some(guidance.into());
        self
    }

    async fn summary(&self, ctx: &HookContext<'_>) -> Option<ConversationSummary> {
        match &self.summary {
            Source::Fixed(summary) => summary.clone(),
            Source::Store(store) => {
                if let Some(pinned) = self
                    .pinned
                    .lock()
                    .ok()
                    .and_then(|p| p.get(&ctx.run_id).cloned())
                {
                    return pinned;
                }
                let summary = store.load(ctx.session_id).await;
                if let Ok(mut pinned) = self.pinned.lock() {
                    pinned.insert(ctx.run_id, summary.clone());
                }
                summary
            }
        }
    }
}

#[async_trait]
impl Hook for SummaryNote {
    async fn before_model(&self, ctx: &HookContext<'_>, request: &mut ModelRequest) {
        if let Some(summary) = self.summary(ctx).await.filter(|s| !s.text.is_empty()) {
            insert_after_system(
                &mut request.messages,
                Message::hook_note(render_summary_note(&summary, self.guidance.as_deref())),
            );
        }
    }

    async fn on_run_end(&self, ctx: &HookContext<'_>, _outcome: &RunOutcome) {
        if let Ok(mut pinned) = self.pinned.lock() {
            pinned.remove(&ctx.run_id);
        }
    }
}

/// A run's transcript as a [`ConversationSource`]: each turn starts at a
/// user message and its `seq` is the offset just past it, so a summary's
/// cursor is the number of leading messages it covers. A fingerprint of
/// those messages detects a transcript that no longer starts with them.
struct Transcript {
    messages: Vec<Message>,
    newest_finished: bool,
}

impl Transcript {
    fn covered(&self, summary: Option<&ConversationSummary>) -> Option<usize> {
        let summary = summary?;
        let covered = usize::try_from(summary.cursor).ok()?;
        (covered <= self.messages.len()
            && fingerprint(&self.messages[..covered]) == summary.fingerprint)
            .then_some(covered)
    }

    fn turns_from(&self, start: usize) -> Vec<ConversationTurn> {
        let starts: Vec<usize> = self
            .messages
            .iter()
            .enumerate()
            .skip(start)
            .filter(|(i, m)| *i == start || m.role == Role::User)
            .map(|(i, _)| i)
            .collect();
        starts
            .iter()
            .enumerate()
            .map(|(n, &from)| {
                let to = starts.get(n + 1).copied().unwrap_or(self.messages.len());
                ConversationTurn::new(to as i64, self.messages[from..to].to_vec())
            })
            .collect()
    }
}

#[async_trait]
impl ConversationSource for Transcript {
    async fn turns_after(
        &self,
        _session_id: &str,
        summary: Option<&ConversationSummary>,
    ) -> Result<PendingTurns, String> {
        let covered = self.covered(summary);
        Ok(PendingTurns {
            turns: self.turns_from(covered.unwrap_or(0)),
            newest_finished: self.newest_finished,
            summary_stale: summary.is_some() && covered.is_none(),
        })
    }

    fn fingerprint(&self, cursor: i64) -> u64 {
        usize::try_from(cursor)
            .ok()
            .filter(|&c| c <= self.messages.len())
            .map_or(0, |c| fingerprint(&self.messages[..c]))
    }
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

/// Keeps a session's transcript inside the model's context window by
/// summarizing older whole turns, the way a long chat should age.
///
/// - **After each run** (`on_run_end`), when the unsummarized transcript
///   passes [`super::compaction_threshold`] of the window, older turns are
///   folded into the session's summary by a background [`Compactor`] pass.
///   Every message is final once a run has ended, so the next run normally
///   finds the work done.
/// - **Before each model call** (`before_model`), the summarized prefix is
///   replaced, in the request only, by one note with the summary
///   ([`render_summary_note`]). The stored transcript is never changed.
///   Only past [`wait_threshold`] does the call wait: for a running pass,
///   else for an inline one, never longer than [`Self::wait_timeout`].
///
/// The window is per model: the request's model name is matched against
/// [`Self::model_window`] entries, else the default. Register it before
/// [`crate::ContextWindow`], which then trims whatever still does not fit.
pub struct ConversationCompaction {
    compactor: Compactor,
    default_window: usize,
    model_windows: Vec<(String, usize)>,
    wait_timeout: Duration,
    guidance: Option<String>,
    /// The model each session last called, for the window after a run.
    session_models: Mutex<HashMap<String, String>>,
}

impl ConversationCompaction {
    /// `default_window` is the context window, in tokens, of models without
    /// a [`Self::model_window`] entry.
    pub fn new(
        summarizer: impl Summarizer + 'static,
        store: impl SummaryStore + 'static,
        default_window: usize,
    ) -> Self {
        Self::with_compactor(Compactor::new(summarizer, store), default_window)
    }

    pub fn with_compactor(compactor: Compactor, default_window: usize) -> Self {
        Self {
            compactor,
            default_window,
            model_windows: Vec::new(),
            wait_timeout: Duration::from_secs(30),
            guidance: None,
            session_models: Mutex::new(HashMap::new()),
        }
    }

    /// The window of models whose name starts with `prefix` (ignoring case
    /// and a provider route such as `openrouter/`); the longest prefix wins.
    pub fn model_window(mut self, prefix: impl Into<String>, tokens: usize) -> Self {
        self.model_windows
            .push((prefix.into().to_ascii_lowercase(), tokens));
        self
    }

    /// The longest a model call waits for compaction (default 30s). After
    /// it, the call goes ahead and [`crate::ContextWindow`] trims.
    pub fn wait_timeout(mut self, timeout: Duration) -> Self {
        self.wait_timeout = timeout;
        self
    }

    /// Appended to the summary note's coverage line.
    pub fn guidance(mut self, guidance: impl Into<String>) -> Self {
        self.guidance = Some(guidance.into());
        self
    }

    pub fn compactor(&self) -> &Compactor {
        &self.compactor
    }

    /// The context window used for `model`.
    pub fn window_for(&self, model: &str) -> usize {
        let model = model.to_ascii_lowercase();
        let name = model.rsplit('/').next().unwrap_or(&model);
        self.model_windows
            .iter()
            .filter(|(prefix, _)| name.starts_with(prefix.as_str()))
            .max_by_key(|(prefix, _)| prefix.len())
            .map_or(self.default_window, |(_, tokens)| *tokens)
    }

    /// The session's current summary.
    pub async fn summary(&self, session_id: &str) -> Option<ConversationSummary> {
        self.compactor.store().load(session_id).await
    }

    /// Wait for every background pass started so far.
    pub async fn drain(&self) {
        self.compactor.jobs().drain().await;
    }

    /// Replace the summarized prefix of the transcript inside `request`.
    /// Leaves the request alone when it does not carry the transcript as
    /// expected (another hook reshaped it).
    fn apply(
        &self,
        request: &mut ModelRequest,
        transcript: &[Message],
        covered: usize,
        note: String,
    ) {
        if covered == 0 || transcript.is_empty() {
            return;
        }
        let Some(start) = request.messages.iter().position(|m| *m == transcript[0]) else {
            return;
        };
        let end = start + covered;
        if request.messages.len() < end || request.messages[start..end] != transcript[..covered] {
            return;
        }
        request
            .messages
            .splice(start..end, [Message::hook_note(note)]);
    }
}

#[async_trait]
impl Hook for ConversationCompaction {
    async fn before_model(&self, ctx: &HookContext<'_>, request: &mut ModelRequest) {
        if let Ok(mut models) = self.session_models.lock() {
            models.insert(ctx.session_id.to_string(), request.model.clone());
        }
        let window = self.window_for(&request.model);
        let source = Transcript {
            messages: ctx.messages.to_vec(),
            newest_finished: false,
        };
        let store = self.compactor.store();
        let unsummarized = |summary: &Option<ConversationSummary>| {
            let covered = source.covered(summary.as_ref()).unwrap_or(0);
            estimate_tokens(&source.messages[covered..])
                + summary
                    .as_ref()
                    .map_or(0, |s| s.text.chars().count().div_ceil(4))
        };
        let mut summary = store.load(ctx.session_id).await;
        if unsummarized(&summary) > wait_threshold(window) {
            // Near the window: let a running pass finish, else compact now.
            let wait = async {
                if self.compactor.jobs().is_running(ctx.session_id) {
                    self.compactor.jobs().wait(ctx.session_id).await;
                } else if let Err(error) = self
                    .compactor
                    .compact(
                        ctx.session_id,
                        &source,
                        CompactionLimits::for_window(window),
                    )
                    .await
                {
                    tracing::warn!(session_id = ctx.session_id, %error, "conversation compaction failed");
                }
            };
            if tokio::time::timeout(self.wait_timeout, wait).await.is_err() {
                tracing::warn!(
                    session_id = ctx.session_id,
                    "conversation compaction did not finish in time; sending the request as is"
                );
            }
            summary = store.load(ctx.session_id).await;
        }
        let Some(summary) = summary else {
            return;
        };
        if let Some(covered) = source.covered(Some(&summary)) {
            let note = render_summary_note(&summary, self.guidance.as_deref());
            self.apply(request, &source.messages, covered, note);
        }
    }

    async fn on_run_end(&self, ctx: &HookContext<'_>, _outcome: &RunOutcome) {
        let model = self
            .session_models
            .lock()
            .ok()
            .and_then(|models| models.get(ctx.session_id).cloned())
            .unwrap_or_default();
        self.compactor.spawn(
            ctx.session_id,
            Arc::new(Transcript {
                messages: ctx.messages.to_vec(),
                newest_finished: true,
            }),
            CompactionLimits::for_window(self.window_for(&model)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::InMemorySummaries;
    use aidaemon_core::ToolCall;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone)]
    struct Counting(Arc<AtomicUsize>, Duration);

    #[async_trait]
    impl Summarizer for Counting {
        async fn summarize(
            &self,
            _: Option<&str>,
            _: &[ConversationTurn],
        ) -> Result<String, String> {
            tokio::time::sleep(self.1).await;
            let n = self.0.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(format!("folded #{n}"))
        }
    }

    fn counting(delay: Duration) -> (Counting, Arc<AtomicUsize>) {
        let count = Arc::new(AtomicUsize::new(0));
        (Counting(count.clone(), delay), count)
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
            ..Default::default()
        }
    }

    fn ctx<'a>(session_id: &'a str, messages: &'a [Message]) -> HookContext<'a> {
        HookContext {
            run_id: RunId::new(),
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

    fn outcome() -> RunOutcome {
        RunOutcome {
            status: aidaemon_core::RunStatus::Completed,
            final_text: None,
            state: aidaemon_core::RunState::new("agent", "s", "hi"),
        }
    }

    fn summary(text: &str) -> ConversationSummary {
        ConversationSummary {
            text: text.into(),
            cursor: 12,
            message_count: 30,
            last_message_id: Some("m-30".into()),
            fingerprint: 0,
        }
    }

    #[test]
    fn transcript_turns_start_at_user_messages() {
        let source = Transcript {
            messages: transcript(3, 100),
            newest_finished: true,
        };
        let seqs: Vec<i64> = source.turns_from(0).iter().map(|t| t.seq).collect();
        assert_eq!(seqs, vec![4, 8, 12]);
        assert_eq!(source.turns_from(4)[0].messages[0].role, Role::User);
        assert!(source.turns_from(12).is_empty());
    }

    #[test]
    fn window_follows_the_longest_matching_model_prefix() {
        let (summarizer, _) = counting(Duration::ZERO);
        let c = ConversationCompaction::new(summarizer, InMemorySummaries::default(), 128_000)
            .model_window("gpt-6", 272_000)
            .model_window("gpt-6-mini", 64_000)
            .model_window("claude-", 200_000);
        assert_eq!(c.window_for("gpt-6-sol"), 272_000);
        assert_eq!(c.window_for("GPT-6-mini-2026"), 64_000);
        assert_eq!(c.window_for("openrouter/claude-sonnet-5-5"), 200_000);
        assert_eq!(c.window_for("local-model"), 128_000);
    }

    #[test]
    fn the_note_names_its_coverage_and_carries_the_state() {
        let note = render_summary_note(&summary("## Active goals"), Some("Use search_history."));
        assert!(note.starts_with("[Context Coverage]\nCompacted through turn 12 / message m-30."));
        assert!(
            note.contains("Use search_history.\n\n[Compacted Conversation State]\n## Active goals")
        );
    }

    /// The pass after a run leaves the summary ready; the next run's model
    /// call swaps it in without summarizing again.
    #[tokio::test]
    async fn run_end_compacts_and_the_next_call_uses_it() {
        let (summarizer, count) = counting(Duration::ZERO);
        let c = ConversationCompaction::new(summarizer, InMemorySummaries::default(), 10_000);
        let t = transcript(8, 1_000);
        c.on_run_end(&ctx("s", &t), &outcome()).await;
        c.drain().await;
        assert_eq!(count.load(Ordering::SeqCst), 1);
        let stored = c.summary("s").await.unwrap();
        assert_eq!(t[stored.cursor as usize].role, Role::User);

        let mut next = t.clone();
        next.push(Message::user("Next question"));
        let mut req = request("m", &next);
        c.before_model(&ctx("s", &next), &mut req).await;
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(req.messages[0].content, "You are a helpful agent.");
        assert!(req.messages[1].is_hook_note());
        assert!(req.messages[1].content.contains("folded #1"));
        assert_eq!(req.messages[2], next[stored.cursor as usize]);
        assert_eq!(req.messages.len(), 2 + next.len() - stored.cursor as usize);
    }

    #[tokio::test]
    async fn below_the_threshold_nothing_is_summarized() {
        let (summarizer, count) = counting(Duration::ZERO);
        let c = ConversationCompaction::new(summarizer, InMemorySummaries::default(), 100_000);
        let t = transcript(20, 2_500); // 50% of the window
        c.on_run_end(&ctx("s", &t), &outcome()).await;
        c.drain().await;
        assert_eq!(count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn the_same_history_compacts_for_a_small_window_only() {
        let t = transcript(10, 1_000);
        let (summarizer, count) = counting(Duration::ZERO);
        let c = ConversationCompaction::new(summarizer, InMemorySummaries::default(), 200_000)
            .model_window("small", 12_000);
        let mut req = request("big-model", &t);
        c.before_model(&ctx("big", &t), &mut req).await;
        c.on_run_end(&ctx("big", &t), &outcome()).await;
        let mut req = request("small-model", &t);
        c.before_model(&ctx("small", &t), &mut req).await;
        c.on_run_end(&ctx("small", &t), &outcome()).await;
        c.drain().await;
        assert!(c.summary("big").await.is_none());
        assert!(c.summary("small").await.is_some());
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_different_history_does_not_use_the_summary() {
        let (summarizer, _) = counting(Duration::ZERO);
        let c = ConversationCompaction::new(summarizer, InMemorySummaries::default(), 10_000);
        let t = transcript(8, 1_000);
        c.on_run_end(&ctx("s", &t), &outcome()).await;
        c.drain().await;
        let mut other = transcript(2, 10);
        other[0] = Message::user("A different conversation");
        let mut req = request("m", &other);
        c.before_model(&ctx("s", &other), &mut req).await;
        assert_eq!(req.messages.len(), 1 + other.len());
        assert!(!req.messages.iter().any(Message::is_hook_note));
    }

    /// Between 60% and 80% a model call never waits.
    #[tokio::test]
    async fn below_the_wait_threshold_a_call_never_summarizes() {
        let (summarizer, count) = counting(Duration::ZERO);
        let c = ConversationCompaction::new(summarizer, InMemorySummaries::default(), 10_000);
        let t = transcript(7, 1_000);
        let mut req = request("m", &t);
        c.before_model(&ctx("s", &t), &mut req).await;
        assert_eq!(count.load(Ordering::SeqCst), 0);
        assert_eq!(req.messages.len(), 1 + t.len());
    }

    #[tokio::test]
    async fn near_the_window_a_call_waits_for_the_running_pass() {
        let (summarizer, count) = counting(Duration::from_millis(50));
        let c = ConversationCompaction::new(summarizer, InMemorySummaries::default(), 10_000);
        let t = transcript(10, 1_000);
        c.on_run_end(&ctx("s", &t), &outcome()).await;
        let mut req = request("m", &t);
        c.before_model(&ctx("s", &t), &mut req).await;
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert!(req.messages[1].content.contains("folded #1"));
        assert!(req.messages.len() < 1 + t.len());
    }

    #[tokio::test]
    async fn near_the_window_with_nothing_running_the_call_compacts() {
        let (summarizer, count) = counting(Duration::ZERO);
        let c = ConversationCompaction::new(summarizer, InMemorySummaries::default(), 200_000)
            .model_window("small", 10_000);
        let t = transcript(10, 1_000);
        let mut req = request("small-model", &t);
        c.before_model(&ctx("s", &t), &mut req).await;
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert!(req.messages[1].content.contains("folded #1"));
    }

    #[tokio::test]
    async fn a_slow_summary_does_not_hold_the_call_past_the_timeout() {
        let (summarizer, _) = counting(Duration::from_secs(60));
        let c = ConversationCompaction::new(summarizer, InMemorySummaries::default(), 10_000)
            .wait_timeout(Duration::from_millis(20));
        let t = transcript(10, 1_000);
        let mut req = request("m", &t);
        let started = std::time::Instant::now();
        c.before_model(&ctx("s", &t), &mut req).await;
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(req.messages.len(), 1 + t.len());
    }

    /// Through an agent: each run's end compacts and later requests carry
    /// the summary instead of the oldest turns.
    #[tokio::test]
    async fn an_agent_conversation_ages_into_a_summary() {
        use aidaemon_core::testing::{text, ScriptedProvider};
        use aidaemon_core::Agent;

        let (summarizer, count) = counting(Duration::ZERO);
        let compaction = Arc::new(ConversationCompaction::new(
            summarizer,
            InMemorySummaries::default(),
            3_000,
        ));
        let seen = Arc::new(Mutex::new(Vec::<ModelRequest>::new()));
        struct Capture(Arc<Mutex<Vec<ModelRequest>>>);
        #[async_trait]
        impl Hook for Capture {
            async fn before_model(&self, _ctx: &HookContext<'_>, request: &mut ModelRequest) {
                self.0.lock().unwrap().push(request.clone());
            }
        }
        let long_answer = "detail ".repeat(400);
        let agent = aidaemon_core::Agent::builder("compacting")
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
        let _: &Agent = &agent;

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
        assert!(count.load(Ordering::SeqCst) > 0);
        let last = seen.lock().unwrap().last().cloned().unwrap();
        assert!(last
            .messages
            .iter()
            .any(|m| m.is_hook_note() && m.content.contains("[Compacted Conversation State]")));
        assert_eq!(last.messages.last().unwrap().content, "fourth");
    }

    #[tokio::test]
    async fn the_note_hook_pins_one_summary_per_run() {
        let store: Arc<dyn SummaryStore> = Arc::new(InMemorySummaries::default());
        store.save("s", summary("first state")).await;
        let note = SummaryNote::new(store.clone()).guidance("Use search_history.");
        let t = vec![Message::user("hello")];
        let run = ctx("s", &t);
        let mut req = request("m", &t);
        note.before_model(&run, &mut req).await;
        assert!(req.messages[1].is_hook_note());
        assert!(req.messages[1].content.contains("first state"));
        // A newer summary mid-run does not change this run's note.
        store.save("s", summary("second state")).await;
        let mut again = request("m", &t);
        note.before_model(&run, &mut again).await;
        assert!(again.messages[1].content.contains("first state"));
        note.on_run_end(&run, &outcome()).await;
        let mut next = request("m", &t);
        note.before_model(&ctx("s", &t), &mut next).await;
        assert!(next.messages[1].content.contains("second state"));
    }

    #[tokio::test]
    async fn a_fixed_note_without_a_summary_adds_nothing() {
        let t = vec![Message::user("hello")];
        let mut req = request("m", &t);
        SummaryNote::fixed(None)
            .before_model(&ctx("s", &t), &mut req)
            .await;
        assert_eq!(req.messages.len(), 2);
    }
}
