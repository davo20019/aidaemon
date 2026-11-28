//! Each session's chat operations: the running turn (cancel token,
//! heartbeat), messages queued behind it, steering messages taken mid-turn
//! and duplicate suppression. Internal to aidaemon, no stability promise.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use aidaemon_events::conversation::MessageAttachment;
use aidaemon_tools::semantics::UserRole;

use crate::context::ChannelContext;

/// Window in seconds during which identical messages to the same session
/// are treated as duplicates, even after the original was popped from the queue.
const DEDUP_WINDOW_SECS: i64 = 120;

/// A task still marked `Running` with no sign of life for this many seconds
/// is treated as leaked: its channel handler never finalized it (an abnormal
/// turn exit, a dropped future, or a panic), so without this it would block
/// the session's message queue forever. A task's age says nothing, because a
/// turn may legitimately run for hours: a live turn's heartbeat ticks every
/// few seconds, and only a task without one is judged by when it started.
const MAX_RUNNING_TASK_AGE_SECS: i64 = 3600;

/// A queued message waiting to be processed.
///
/// Serializable so the queue outlives a daemon restart ([`QueueSnapshot`]):
/// a message the daemon accepted is never lost to one.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct QueuedMessage {
    /// Its id while it waits: what an app names to edit or remove it. A
    /// restored message gets a new one ([`TaskRegistry::restore_queues`]).
    pub id: u64,
    pub text: String,
    /// Authorization is captured when the message arrives. Channel workers
    /// must never reuse the role of the message that happened to start the
    /// session worker for later queued messages.
    #[serde(with = "crate::context::role_serde")]
    pub user_role: UserRole,
    /// Full per-message channel/sender context, including any scoped grant.
    pub channel_ctx: ChannelContext,
    /// Attachments belong to the individual sender/message and must travel
    /// through the queue with that identity.
    pub attachments: Vec<MessageAttachment>,
    pub queued_at: DateTime<Utc>,
    /// Optional transport timing follows the exact message through queue
    /// coalescing and later agent dispatch. Not kept across a restart: the
    /// transport's timing says nothing about a turn run after one.
    #[serde(skip)]
    pub ingress_timing: Option<crate::ports::InboundMessageTiming>,
    /// Whether the running turn may take it in mid-run. A message read as
    /// unrelated to the running work waits for its own turn.
    pub steerable: bool,
    /// The owner sent it to wait for its own turn (an app's Queue): the
    /// running work takes messages sent after it to be read now past it.
    pub placed_for_later: bool,
    /// The owner placed it from an app (its Now or After), which lists it
    /// while it waits and shows it in the chat once it runs: the channel
    /// says nothing about it.
    pub listed_by_client: bool,
}

impl QueuedMessage {
    pub fn new(
        text: impl Into<String>,
        user_role: UserRole,
        channel_ctx: ChannelContext,
        attachments: Vec<MessageAttachment>,
    ) -> Self {
        Self {
            id: next_queued_id(),
            text: text.into(),
            user_role,
            channel_ctx,
            attachments,
            queued_at: Utc::now(),
            ingress_timing: None,
            steerable: true,
            placed_for_later: false,
            listed_by_client: false,
        }
    }

    /// Placed by the owner from an app that lists it (see
    /// [`Self::listed_by_client`]).
    pub fn listed_by_client(mut self) -> Self {
        self.listed_by_client = true;
        self
    }

    /// Keep it out of the running turn: it runs as its own turn.
    pub fn not_steerable(mut self) -> Self {
        self.steerable = false;
        self
    }

    /// The owner chose to send it after the running work (an app's Queue).
    pub fn placed_for_later(mut self) -> Self {
        self.steerable = false;
        self.placed_for_later = true;
        self
    }

    pub fn with_ingress_timing(mut self, mut timing: crate::ports::InboundMessageTiming) -> Self {
        timing.mark_queued(self.queued_at);
        self.ingress_timing = Some(timing);
        self
    }

    /// Same sender and context, and the same placement: a message meant
    /// for the running work never runs merged into one the owner queued
    /// for its own turn. And none the owner queued for its own turn joins
    /// another: each was sent as a request of its own (fragments of one
    /// long message, which this merging is for, are not placed).
    fn can_coalesce_with(&self, other: &Self) -> bool {
        self.user_role == other.user_role
            && self.channel_ctx == other.channel_ctx
            && self.steerable == other.steerable
            && !self.placed_for_later
            && !other.placed_for_later
    }
}

/// A new id for a waiting message, unique in this process.
fn next_queued_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Every session's waiting messages, oldest first: what the registry
/// journals on each change ([`TaskRegistry::journal_to`]) and restores at
/// boot ([`TaskRegistry::restore_queues`]).
pub type QueueSnapshot = std::collections::BTreeMap<String, Vec<QueuedMessage>>;

/// Outcome of an atomic check-and-queue attempt.
#[derive(Debug)]
pub enum QueueOutcome {
    /// Message queued at this 1-based position.
    Queued(usize),
    /// Identical message recently seen — silently dropped.
    Duplicate,
    /// No running task for the session: queueing would strand the message
    /// (nothing will drain it). The caller must process it directly.
    NoRunningTask,
}

#[derive(Clone, Debug)]
pub enum TaskStatus {
    Running,
    /// Cancellation was requested, but the owning channel worker has not yet
    /// durably terminalized the agent task or handed off the session FIFO.
    Cancelling,
    Completed,
    Failed(String),
    Cancelled,
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskStatus::Running => write!(f, "Running"),
            TaskStatus::Cancelling => write!(f, "Cancelling"),
            TaskStatus::Completed => write!(f, "Completed"),
            TaskStatus::Failed(e) => write!(f, "Failed: {}", e),
            TaskStatus::Cancelled => write!(f, "Cancelled"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TaskEntry {
    pub id: u64,
    pub session_id: String,
    pub description: String,
    pub status: TaskStatus,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
}

struct TaskHandle {
    entry: TaskEntry,
    cancel_token: CancellationToken,
    /// Optional token to cancel the channel's typing indicator for this task.
    /// Set by the channel after task registration; cancelled alongside the task.
    typing_cancel: Option<CancellationToken>,
    /// Unix seconds of the turn's last sign of life, once it has one.
    heartbeat: Option<Arc<AtomicU64>>,
}

impl TaskHandle {
    fn seconds_since_alive(&self, now: DateTime<Utc>) -> i64 {
        match &self.heartbeat {
            Some(heartbeat) => now.timestamp() - heartbeat.load(Ordering::Relaxed) as i64,
            None => (now - self.entry.started_at).num_seconds(),
        }
    }
}

pub struct TaskRegistry {
    tasks: RwLock<HashMap<u64, TaskHandle>>,
    next_id: AtomicU64,
    max_completed: usize,
    /// Message queues per session - messages wait here when a task is running.
    queues: RwLock<HashMap<String, VecDeque<QueuedMessage>>>,
    /// Recently seen message fingerprints per session for deduplication.
    /// Key: (session_id, text_hash), Value: timestamp when first seen.
    /// This survives message pops so webhook retries are caught.
    recently_seen: RwLock<HashMap<(String, u64), DateTime<Utc>>>,
    /// The short numbers `/tasks` shows for each chat's sub-agents and
    /// background commands (by their agent key), drawn from the same counter
    /// as turn numbers so one `/cancel <n>` names either.
    work_numbers: std::sync::Mutex<HashMap<String, Vec<(String, u64)>>>,
    /// When each chat last asked for `/wipe`, awaiting its confirmation.
    wipe_requests: std::sync::Mutex<HashMap<String, std::time::Instant>>,
    /// Where each change to the queues is published, latest first wins
    /// ([`Self::journal_to`]); unset, the queues live in memory only.
    journal: std::sync::OnceLock<tokio::sync::watch::Sender<QueueSnapshot>>,
}

/// How long a `/wipe` waits for the second `/wipe` that confirms it.
pub const WIPE_CONFIRM_WINDOW: std::time::Duration = std::time::Duration::from_secs(120);

/// What a `/wipe` in a chat amounts to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WipeRequest {
    /// The first: nothing is erased until it is confirmed.
    Requested,
    /// A second `/wipe` within [`WIPE_CONFIRM_WINDOW`] of the first.
    Confirmed,
}

impl TaskRegistry {
    fn is_active(status: &TaskStatus) -> bool {
        matches!(status, TaskStatus::Running | TaskStatus::Cancelling)
    }

    pub fn new(max_completed: usize) -> Self {
        Self {
            tasks: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            max_completed,
            queues: RwLock::new(HashMap::new()),
            recently_seen: RwLock::new(HashMap::new()),
            work_numbers: std::sync::Mutex::new(HashMap::new()),
            wipe_requests: std::sync::Mutex::new(HashMap::new()),
            journal: std::sync::OnceLock::new(),
        }
    }

    /// Publish every change to the queues on `journal` (a snapshot of all
    /// of them, taken while the change holds the queues' lock, so the
    /// latest value is always the current queues). The daemon stores it,
    /// and a message accepted to wait outlives a restart. Set once.
    pub fn journal_to(&self, journal: tokio::sync::watch::Sender<QueueSnapshot>) {
        let _ = self.journal.set(journal);
    }

    /// Put messages a previous run of the daemon left waiting back on their
    /// sessions' queues, after anything queued since, each with a new id.
    /// Returns the sessions that now have messages waiting: nothing is
    /// running to drain them, so the caller starts their turns.
    pub async fn restore_queues(&self, snapshot: QueueSnapshot) -> Vec<String> {
        let mut queues = self.queues.write().await;
        let mut restored = Vec::new();
        for (session_id, messages) in snapshot {
            if messages.is_empty() {
                continue;
            }
            let queue = queues.entry(session_id.clone()).or_default();
            for mut message in messages {
                message.id = next_queued_id();
                queue.push_back(message);
            }
            restored.push(session_id);
        }
        self.journal(&queues);
        restored
    }

    /// Publish the queues as they are now (`queues` is the locked map).
    fn journal(&self, queues: &HashMap<String, VecDeque<QueuedMessage>>) {
        let Some(journal) = self.journal.get() else {
            return;
        };
        let snapshot: QueueSnapshot = queues
            .iter()
            .filter(|(_, queue)| !queue.is_empty())
            .map(|(session, queue)| (session.clone(), queue.iter().cloned().collect()))
            .collect();
        journal.send_if_modified(|current| {
            let changed = !same_snapshot(current, &snapshot);
            if changed {
                *current = snapshot;
            }
            changed
        });
    }

    /// Record a `/wipe` in `session_id` at `now`. The second one within
    /// [`WIPE_CONFIRM_WINDOW`] confirms (and consumes) the first; a later
    /// one starts over. A confirmation belongs to its own chat only.
    pub fn request_wipe(&self, session_id: &str, now: std::time::Instant) -> WipeRequest {
        let mut requests = self
            .wipe_requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        requests.retain(|_, at| now.saturating_duration_since(*at) <= WIPE_CONFIRM_WINDOW);
        if requests.remove(session_id).is_some() {
            WipeRequest::Confirmed
        } else {
            requests.insert(session_id.to_string(), now);
            WipeRequest::Requested
        }
    }

    /// Simple hash for dedup fingerprinting.
    fn text_hash(text: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// Register a new task. Returns the task ID and a cancellation token.
    pub async fn register(&self, session_id: &str, description: &str) -> (u64, CancellationToken) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let cancel_token = CancellationToken::new();
        let handle = TaskHandle {
            entry: TaskEntry {
                id,
                session_id: session_id.to_string(),
                description: description.to_string(),
                status: TaskStatus::Running,
                started_at: Utc::now(),
                finished_at: None,
            },
            cancel_token: cancel_token.clone(),
            typing_cancel: None,
            heartbeat: None,
        };
        let mut tasks = self.tasks.write().await;
        tasks.insert(id, handle);
        (id, cancel_token)
    }

    /// Mark a task as completed.
    /// Channel loops now use [`Self::finalize_and_drain`]; retained for
    /// callers that finalize without a session queue (and for tests).
    pub async fn complete(&self, task_id: u64) {
        let mut tasks = self.tasks.write().await;
        if let Some(handle) = tasks.get_mut(&task_id) {
            handle.entry.status = TaskStatus::Completed;
            handle.entry.finished_at = Some(Utc::now());
        }
        Self::cleanup_locked(&mut tasks, self.max_completed);
    }

    /// Mark a task as failed.
    pub async fn fail(&self, task_id: u64, error: &str) {
        let mut tasks = self.tasks.write().await;
        if let Some(handle) = tasks.get_mut(&task_id) {
            handle.entry.status = TaskStatus::Failed(error.to_string());
            handle.entry.finished_at = Some(Utc::now());
        }
        Self::cleanup_locked(&mut tasks, self.max_completed);
    }

    /// Give a running task the heartbeat that shows it is alive.
    pub async fn set_heartbeat(&self, task_id: u64, heartbeat: Arc<AtomicU64>) {
        let mut tasks = self.tasks.write().await;
        if let Some(handle) = tasks.get_mut(&task_id) {
            handle.heartbeat = Some(heartbeat);
        }
    }

    /// Associate a typing indicator cancel token with a running task.
    /// When the task is cancelled, the typing indicator will also be stopped.
    pub async fn set_typing_cancel(&self, task_id: u64, token: CancellationToken) {
        let mut tasks = self.tasks.write().await;
        if let Some(handle) = tasks.get_mut(&task_id) {
            handle.typing_cancel = Some(token);
        }
    }

    /// Cancel a running task. Returns true if the task was found and cancelled.
    pub async fn cancel(&self, task_id: u64) -> bool {
        let mut tasks = self.tasks.write().await;
        if let Some(handle) = tasks.get_mut(&task_id)
            && matches!(handle.entry.status, TaskStatus::Running)
        {
            handle.cancel_token.cancel();
            if let Some(ref typing) = handle.typing_cancel {
                typing.cancel();
            }
            handle.entry.status = TaskStatus::Cancelling;
            return true;
        }
        false
    }

    /// Cancel all running tasks for a session. Returns cancelled task (ID, description) pairs.
    pub async fn cancel_running_for_session(&self, session_id: &str) -> Vec<(u64, String)> {
        let mut tasks = self.tasks.write().await;
        let mut cancelled = Vec::new();
        for (id, handle) in tasks.iter_mut() {
            if handle.entry.session_id == session_id
                && matches!(handle.entry.status, TaskStatus::Running)
            {
                handle.cancel_token.cancel();
                if let Some(ref typing) = handle.typing_cancel {
                    typing.cancel();
                }
                handle.entry.status = TaskStatus::Cancelling;
                cancelled.push((*id, handle.entry.description.clone()));
            }
        }
        cancelled
    }

    /// The numbers of the chat's running work `keys`, in order: a key keeps
    /// its number while it runs, a new one gets the next free number, and
    /// keys no longer running are forgotten.
    pub fn number_work(&self, session_id: &str, keys: &[&str]) -> Vec<u64> {
        let mut numbers = self
            .work_numbers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let known = numbers.remove(session_id).unwrap_or_default();
        let mut kept: Vec<(String, u64)> = Vec::with_capacity(keys.len());
        let assigned = keys
            .iter()
            .map(|key| {
                let number = known
                    .iter()
                    .find(|(known_key, _)| known_key == key)
                    .map(|(_, number)| *number)
                    .unwrap_or_else(|| self.next_id.fetch_add(1, Ordering::Relaxed));
                kept.push((key.to_string(), number));
                number
            })
            .collect();
        if !kept.is_empty() {
            numbers.insert(session_id.to_string(), kept);
        }
        assigned
    }

    /// The turn `task_id` as the registry holds it now.
    pub async fn entry(&self, task_id: u64) -> Option<TaskEntry> {
        self.tasks
            .read()
            .await
            .get(&task_id)
            .map(|handle| handle.entry.clone())
    }

    /// List all tasks for a given session, sorted by ID.
    pub async fn list_for_session(&self, session_id: &str) -> Vec<TaskEntry> {
        let tasks = self.tasks.read().await;
        let mut entries: Vec<TaskEntry> = tasks
            .values()
            .filter(|h| h.entry.session_id == session_id)
            .map(|h| h.entry.clone())
            .collect();
        entries.sort_by_key(|e| e.id);
        entries
    }

    /// Mark leaked `Running` tasks (older than [`MAX_RUNNING_TASK_AGE_SECS`]) as
    /// `Failed` so they stop blocking the session queue and surface as resolved
    /// in `/tasks`. Cancels each reaped task's token to nudge any still-alive
    /// hung future to exit. Returns the reaped task ids for logging.
    fn reap_stale_running_locked(tasks: &mut HashMap<u64, TaskHandle>) -> Vec<u64> {
        let now = Utc::now();
        let mut reaped = Vec::new();
        for handle in tasks.values_mut() {
            let stale = Self::is_active(&handle.entry.status)
                && handle.seconds_since_alive(now) > MAX_RUNNING_TASK_AGE_SECS;
            if stale {
                handle.cancel_token.cancel();
                if let Some(ref typing) = handle.typing_cancel {
                    typing.cancel();
                }
                handle.entry.status =
                    TaskStatus::Failed("stale: task was never finalized".to_string());
                handle.entry.finished_at = Some(now);
                reaped.push(handle.entry.id);
            }
        }
        reaped
    }

    /// Remove oldest finished tasks when count exceeds max_completed.
    fn cleanup_locked(tasks: &mut HashMap<u64, TaskHandle>, max_completed: usize) {
        let mut finished: Vec<u64> = tasks
            .iter()
            .filter(|(_, h)| !Self::is_active(&h.entry.status))
            .map(|(&id, _)| id)
            .collect();

        if finished.len() <= max_completed {
            return;
        }

        // Sort ascending by ID (oldest first) and remove excess
        finished.sort();
        let to_remove = finished.len() - max_completed;
        for &id in finished.iter().take(to_remove) {
            tasks.remove(&id);
        }
    }

    /// Check if a session has a running task.
    ///
    /// Reaps leaked (stale) running tasks first, so a phantom task that was
    /// never finalized can't make this return `true` forever and block the
    /// session's queue.
    pub async fn has_running_task(&self, session_id: &str) -> bool {
        let mut tasks = self.tasks.write().await;
        let reaped = Self::reap_stale_running_locked(&mut tasks);
        if !reaped.is_empty() {
            tracing::warn!(
                ?reaped,
                session_id,
                "Reaped stale running task(s) that were never finalized"
            );
        }
        tasks
            .values()
            .any(|h| h.entry.session_id == session_id && Self::is_active(&h.entry.status))
    }

    /// Atomically mark a message as "seen" for dedup purposes.
    /// Returns `true` if this is the first time (caller should proceed),
    /// or `false` if a duplicate (caller should drop the message).
    ///
    /// Call this at the start of **direct** message processing (when no task
    /// is already running) to prevent concurrent webhook handlers from all
    /// bypassing `queue_message()` and starting separate processing flows.
    ///
    /// `dedup_key` is the channel's stable per-message identity (e.g. Telegram
    /// `message_id`, Slack `ts`, Discord message id) — the correct idempotency
    /// key for webhook/poll redeliveries. When `Some`, dedup is keyed on it so a
    /// user *re-typing the same text* (which gets a NEW message id) is treated
    /// as a fresh request rather than a duplicate. Falls back to hashing `text`
    /// when `None`, preserving the legacy content-based behavior.
    pub async fn mark_seen(&self, session_id: &str, text: &str, dedup_key: Option<&str>) -> bool {
        let now = Utc::now();
        let hash = Self::text_hash(dedup_key.unwrap_or(text));
        let key = (session_id.to_string(), hash);
        let mut seen = self.recently_seen.write().await;
        seen.retain(|_, ts| (now - *ts).num_seconds() < DEDUP_WINDOW_SECS);
        use std::collections::hash_map::Entry;
        match seen.entry(key) {
            Entry::Occupied(_) => false,
            Entry::Vacant(e) => {
                e.insert(now);
                true
            }
        }
    }

    /// Undo [`Self::mark_seen`] for a message handled in two steps (a stop
    /// done now, the rest of the message then run or queued), so its second
    /// step is not taken for a duplicate delivery.
    pub async fn unmark_seen(&self, session_id: &str, text: &str, dedup_key: Option<&str>) {
        let hash = Self::text_hash(dedup_key.unwrap_or(text));
        self.recently_seen
            .write()
            .await
            .remove(&(session_id.to_string(), hash));
    }

    /// Queue a message for later processing.
    /// Returns `Some(position)` if queued, or `None` if deduplicated (identical message
    /// already exists in the queue or was recently processed).
    /// Channel loops now use [`Self::queue_message_if_running`]; retained
    /// for queue-only callers and tests.
    pub async fn queue_message(
        &self,
        session_id: &str,
        message: QueuedMessage,
        dedup_key: Option<&str>,
    ) -> Option<usize> {
        let now = Utc::now();
        let hash = Self::text_hash(dedup_key.unwrap_or(&message.text));
        let key = (session_id.to_string(), hash);

        // Check recently_seen first — this survives queue pops and catches
        // webhook retries even after the original message was processed.
        {
            let mut seen = self.recently_seen.write().await;
            // Prune stale entries while we hold the lock.
            seen.retain(|_, ts| (now - *ts).num_seconds() < DEDUP_WINDOW_SECS);

            if let Some(first_seen) = seen.get(&key)
                && (now - *first_seen).num_seconds() < DEDUP_WINDOW_SECS
            {
                return None; // Duplicate — silently drop
            }
            seen.insert(key, now);
        }

        let mut queues = self.queues.write().await;
        let queue = queues.entry(session_id.to_string()).or_default();

        queue.push_back(message);
        let position = queue.len();
        self.journal(&queues);
        Some(position)
    }

    /// Pop the next queued message for a session.
    pub async fn pop_queued_message(&self, session_id: &str) -> Option<QueuedMessage> {
        let mut queues = self.queues.write().await;
        let next = queues.get_mut(session_id).and_then(|q| q.pop_front());
        self.journal(&queues);
        next
    }

    /// Pop the next queued message and coalesce only immediately-adjacent
    /// fragments from the exact same authorization and channel context.
    /// Messages from different senders/scopes stay queued for a separate turn;
    /// otherwise a group participant could inherit another participant's role.
    fn drain_and_coalesce(queue: &mut VecDeque<QueuedMessage>) -> Option<QueuedMessage> {
        // A message meant for the work that just ended runs before ones the
        // same sender placed for later: they asked for it first in effect.
        // Another sender's message, or anything not placed for later, keeps
        // the order.
        let start = match queue.iter().position(|m| !m.placed_for_later) {
            Some(at)
                if at > 0
                    && queue.iter().take(at).all(|m| {
                        m.user_role == queue[at].user_role && m.channel_ctx == queue[at].channel_ctx
                    }) =>
            {
                at
            }
            _ => 0,
        };
        let mut combined = queue.remove(start)?;
        while queue
            .get(start)
            .is_some_and(|next| combined.can_coalesce_with(next))
        {
            let msg = queue.remove(start).expect("checked");
            combined.text.push('\n');
            combined.text.push_str(&msg.text);
            combined.attachments.extend(msg.attachments);
            combined.listed_by_client &= msg.listed_by_client;
        }
        Some(combined)
    }

    /// Pop the next same-principal cohort for a session and coalesce its
    /// adjacent fragments. Different security contexts remain queued.
    /// Channel loops now use [`Self::finalize_and_drain`]; retained for
    /// drain-only callers and tests.
    pub async fn pop_all_queued_messages(&self, session_id: &str) -> Option<QueuedMessage> {
        let mut queues = self.queues.write().await;
        let next = Self::drain_and_coalesce(queues.get_mut(session_id)?);
        self.journal(&queues);
        next
    }

    /// Take the queued messages a running turn can absorb mid-run: from the
    /// front of the queue, while each comes from the same principal and
    /// channel context as the turn (`user_role`, `channel_ctx`), carries
    /// no attachments and was not kept for its own turn (`steerable`). The
    /// first message that differs, and everything after it, stays queued for
    /// its own turn, so order and authorization hold.
    pub async fn take_steering_messages(
        &self,
        session_id: &str,
        user_role: UserRole,
        channel_ctx: &ChannelContext,
    ) -> Vec<QueuedMessage> {
        let mut queues = self.queues.write().await;
        let Some(queue) = queues.get_mut(session_id) else {
            return Vec::new();
        };
        // From the front, while each comes from the turn's principal and
        // context: a message meant for the running work is taken; one the
        // owner placed to wait for its own turn keeps its place and does not
        // hold back what they sent after it to be read now (live
        // 2026-11-15: "also this", sent with Enter behind "after this" sent
        // with Queue, ran merged into the queued turn). Anything else (another
        // sender, files, a message read as unrelated) stops the take, so
        // order and authorization hold.
        let mut taken = Vec::new();
        let mut kept = VecDeque::with_capacity(queue.len());
        let mut stopped = false;
        for next in queue.drain(..) {
            // A sending device's ceiling does not make it another sender:
            // the running turn takes the ceiling on with the message.
            let same = next.user_role == user_role && next.channel_ctx.same_sender_as(channel_ctx);
            if !stopped && same && next.steerable && next.attachments.is_empty() {
                taken.push(next);
            } else {
                if !(same && next.placed_for_later) {
                    stopped = true;
                }
                kept.push_back(next);
            }
        }
        *queue = kept;
        self.journal(&queues);
        taken
    }

    /// Put messages a running turn took (`take_steering_messages`) but
    /// never answered back at the front of the session's queue, in their
    /// original order, so the channel runs them as their own turn once the
    /// current one ends. The turn calls this before it returns, so the
    /// channel's `finalize_and_drain` still finds them.
    pub async fn return_steering_messages(&self, session_id: &str, messages: Vec<QueuedMessage>) {
        if messages.is_empty() {
            return;
        }
        let mut queues = self.queues.write().await;
        let queue = queues.entry(session_id.to_string()).or_default();
        for message in messages.into_iter().rev() {
            queue.push_front(message);
        }
        self.journal(&queues);
    }

    /// The messages waiting in `session_id`'s queue, oldest first: id,
    /// text, when each was queued, and whether the owner placed it for
    /// after the running work.
    pub async fn queued_messages(
        &self,
        session_id: &str,
    ) -> Vec<(u64, String, DateTime<Utc>, bool)> {
        self.queues
            .read()
            .await
            .get(session_id)
            .map(|queue| {
                queue
                    .iter()
                    .map(|m| (m.id, m.text.clone(), m.queued_at, m.placed_for_later))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Change the text of queued message `id` (`Some`), or take it out of
    /// the queue (`None`). `false` when it is no longer waiting (it ran).
    pub async fn edit_queued_message(
        &self,
        session_id: &str,
        id: u64,
        text: Option<String>,
    ) -> bool {
        let mut queues = self.queues.write().await;
        let Some(queue) = queues.get_mut(session_id) else {
            return false;
        };
        let Some(at) = queue.iter().position(|m| m.id == id) else {
            return false;
        };
        match text {
            Some(text) => queue[at].text = text,
            None => {
                queue.remove(at);
            }
        }
        self.journal(&queues);
        true
    }

    /// Give queued message `id`, sent to wait for its own turn, to the
    /// running work: it is read at the work's next step, the same as a
    /// message sent to be read now. Nothing running is interrupted. `false`
    /// when it is no longer waiting or was not sent to wait.
    pub async fn send_queued_now(&self, session_id: &str, id: u64) -> bool {
        let mut queues = self.queues.write().await;
        let Some(message) = queues
            .get_mut(session_id)
            .and_then(|queue| queue.iter_mut().find(|m| m.id == id))
        else {
            return false;
        };
        if !message.placed_for_later {
            return false;
        }
        message.placed_for_later = false;
        message.steerable = true;
        self.journal(&queues);
        true
    }

    /// Atomically check for a running task and queue the message behind it.
    ///
    /// Closes the race where a session task finishes between the caller's
    /// `has_running_task()` check and `queue_message()`: a message queued
    /// after the drain in `finalize_and_drain` would otherwise sit in the
    /// queue with nothing left to drain it (observed in the 2026-06-06
    /// attribution run, where a queued message was silently stranded).
    ///
    /// Lock order: `tasks` → `recently_seen` → `queues` (same as
    /// `finalize_and_drain`), so the two operations serialize and the
    /// caller can trust the [`QueueOutcome`].
    pub async fn queue_message_if_running(
        &self,
        session_id: &str,
        message: QueuedMessage,
        dedup_key: Option<&str>,
    ) -> QueueOutcome {
        let mut tasks = self.tasks.write().await;
        // Reap leaked running tasks first: a phantom that was never finalized
        // must not strand this message in a queue nobody will drain.
        let reaped = Self::reap_stale_running_locked(&mut tasks);
        if !reaped.is_empty() {
            tracing::warn!(
                ?reaped,
                session_id,
                "Reaped stale running task(s) before queue decision"
            );
        }
        let running = tasks
            .values()
            .any(|h| h.entry.session_id == session_id && Self::is_active(&h.entry.status));
        if !running {
            return QueueOutcome::NoRunningTask;
        }

        // Keep holding the task-map write lock while queueing so
        // finalize_and_drain (which also takes the write lock first) cannot
        // finalize-and-drain between our check and our push.
        let now = Utc::now();
        let hash = Self::text_hash(dedup_key.unwrap_or(&message.text));
        let key = (session_id.to_string(), hash);
        {
            let mut seen = self.recently_seen.write().await;
            seen.retain(|_, ts| (now - *ts).num_seconds() < DEDUP_WINDOW_SECS);
            if let Some(first_seen) = seen.get(&key)
                && (now - *first_seen).num_seconds() < DEDUP_WINDOW_SECS
            {
                return QueueOutcome::Duplicate;
            }
            seen.insert(key, now);
        }
        let mut queues = self.queues.write().await;
        let queue = queues.entry(session_id.to_string()).or_default();
        queue.push_back(message);
        let position = queue.len();
        self.journal(&queues);
        QueueOutcome::Queued(position)
    }

    /// Atomically finalize a task (complete, or fail with `error`) and drain
    /// the session's message queue in one critical section.
    ///
    /// Holding the task-map write lock across both steps means no concurrent
    /// `queue_message_if_running` can observe the task as running after the
    /// drain — so a message is either drained here or rejected with
    /// `NoRunningTask` (and processed directly by its own handler). Replaces
    /// the racy `complete()`/`fail()` → gap → `pop_all_queued_messages()`
    /// sequence in the channel loops.
    pub async fn finalize_and_drain(
        &self,
        task_id: u64,
        session_id: &str,
        error: Option<&str>,
    ) -> Option<QueuedMessage> {
        let mut tasks = self.tasks.write().await;
        if let Some(handle) = tasks.get_mut(&task_id) {
            // `cancel()` records the authoritative terminal cause before the
            // running future observes its cancellation token. Preserve that
            // state while still using this transaction to hand off the FIFO.
            if matches!(handle.entry.status, TaskStatus::Cancelling) {
                handle.entry.status = TaskStatus::Cancelled;
            } else if !matches!(handle.entry.status, TaskStatus::Cancelled) {
                handle.entry.status = match error {
                    Some(e) => TaskStatus::Failed(e.to_string()),
                    None => TaskStatus::Completed,
                };
            }
            handle.entry.finished_at = Some(Utc::now());
        }
        Self::cleanup_locked(&mut tasks, self.max_completed);

        let mut queues = self.queues.write().await;
        let next = Self::drain_and_coalesce(queues.get_mut(session_id)?);
        self.journal(&queues);
        next
    }

    /// Get the number of queued messages for a session.
    pub async fn queue_len(&self, session_id: &str) -> usize {
        let queues = self.queues.read().await;
        queues.get(session_id).map(|q| q.len()).unwrap_or(0)
    }

    /// Clear all queued messages for a session.
    pub async fn clear_queue(&self, session_id: &str) {
        let mut queues = self.queues.write().await;
        if let Some(queue) = queues.get_mut(session_id) {
            queue.clear();
        }
        self.journal(&queues);
    }
}

/// Whether two snapshots hold the same messages (by id and text, in
/// order): an unchanged queue is not written again.
fn same_snapshot(a: &QueueSnapshot, b: &QueueSnapshot) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|((sa, qa), (sb, qb))| {
            sa == sb
                && qa.len() == qb.len()
                && qa.iter().zip(qb).all(|(ma, mb)| {
                    ma.id == mb.id
                        && ma.text == mb.text
                        && ma.steerable == mb.steerable
                        && ma.placed_for_later == mb.placed_for_later
                })
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wipe_needs_a_second_request_in_the_same_chat_within_the_window() {
        let registry = TaskRegistry::new(10);
        let start = std::time::Instant::now();
        assert_eq!(
            registry.request_wipe("chat-a", start),
            WipeRequest::Requested
        );
        // Another chat's /wipe confirms nothing here.
        assert_eq!(
            registry.request_wipe("chat-b", start),
            WipeRequest::Requested
        );
        let soon = start + std::time::Duration::from_secs(30);
        assert_eq!(
            registry.request_wipe("chat-a", soon),
            WipeRequest::Confirmed
        );
        // Consumed: the next one asks again.
        assert_eq!(
            registry.request_wipe("chat-a", soon),
            WipeRequest::Requested
        );
        // Too late: the old request expired, this one asks again.
        let late = soon + WIPE_CONFIRM_WINDOW + std::time::Duration::from_secs(1);
        assert_eq!(
            registry.request_wipe("chat-a", late),
            WipeRequest::Requested
        );
        assert_eq!(
            registry.request_wipe("chat-b", late),
            WipeRequest::Requested
        );
    }

    #[tokio::test]
    async fn a_long_turn_with_a_live_heartbeat_is_not_reaped() {
        let registry = TaskRegistry::new(10);
        let (live, live_token) = registry.register("live", "long turn").await;
        let (leaked, leaked_token) = registry.register("leaked", "lost turn").await;
        let now = Utc::now().timestamp() as u64;
        registry
            .set_heartbeat(live, Arc::new(AtomicU64::new(now)))
            .await;
        {
            let mut tasks = registry.tasks.write().await;
            let long_ago = Utc::now() - chrono::Duration::hours(6);
            tasks.get_mut(&live).unwrap().entry.started_at = long_ago;
            tasks.get_mut(&leaked).unwrap().entry.started_at = long_ago;
        }

        assert!(registry.has_running_task("live").await);
        assert!(!live_token.is_cancelled());
        assert!(!registry.has_running_task("leaked").await);
        assert!(leaked_token.is_cancelled());
    }

    fn queued(text: &str) -> QueuedMessage {
        queued_as(text, "test:user", UserRole::Public)
    }

    fn queued_as(text: &str, sender_id: &str, user_role: UserRole) -> QueuedMessage {
        let mut channel_ctx = ChannelContext::private("test");
        channel_ctx.sender_id = Some(sender_id.to_string());
        QueuedMessage::new(text, user_role, channel_ctx, Vec::new())
    }

    /// Each change to the queues is published, so the daemon can keep a
    /// message it accepted across a restart; restoring puts the messages
    /// back in order, with their sender's role and context, under new ids.
    #[tokio::test]
    async fn waiting_messages_are_journaled_and_restored_after_a_restart() {
        let registry = TaskRegistry::new(10);
        let (journal, mut saved) = tokio::sync::watch::channel(QueueSnapshot::new());
        registry.journal_to(journal);
        let (task, _) = registry.register("app:s1", "running").await;
        let mut first = queued_as("first", "app:owner", UserRole::Owner);
        first.channel_ctx.in_run = Some(crate::context::RunThread {
            run: "cli-agent-1".into(),
            working_dir: Some("/tmp/project".into()),
            queue_only: true,
        });
        let first = first.placed_for_later();
        registry
            .queue_message_if_running("app:s1", first, Some("r1"))
            .await;
        registry
            .queue_message_if_running(
                "app:s1",
                queued_as("second", "app:owner", UserRole::Owner),
                Some("r2"),
            )
            .await;
        assert!(saved.has_changed().unwrap());
        let stored = serde_json::to_string(&*saved.borrow_and_update()).unwrap();

        // A new process: nothing waits until the stored queues come back.
        let restarted = TaskRegistry::new(10);
        let snapshot: QueueSnapshot = serde_json::from_str(&stored).unwrap();
        assert_eq!(restarted.restore_queues(snapshot).await, vec!["app:s1"]);
        let waiting = restarted.queued_messages("app:s1").await;
        assert_eq!(
            waiting.iter().map(|m| m.1.as_str()).collect::<Vec<_>>(),
            ["first", "second"]
        );
        let next = restarted.pop_queued_message("app:s1").await.unwrap();
        assert_eq!(next.user_role, UserRole::Owner);
        assert!(next.placed_for_later);
        assert_eq!(
            next.channel_ctx.in_run.as_ref().map(|r| r.run.as_str()),
            Some("cli-agent-1")
        );
        assert_ne!(
            waiting[0].0,
            registry.queued_messages("app:s1").await[0].0,
            "a restored message gets an id of this process"
        );

        // Draining publishes the empty queue: nothing comes back next boot.
        registry.finalize_and_drain(task, "app:s1", None).await;
        registry.pop_all_queued_messages("app:s1").await;
        assert!(saved.borrow_and_update().is_empty());
    }

    /// A stored role never reads back as more than it said.
    #[test]
    fn an_unknown_stored_role_reads_back_as_public() {
        let mut message = serde_json::to_value(queued_as("hi", "x", UserRole::Guest)).unwrap();
        assert_eq!(message["user_role"], "guest");
        message["user_role"] = "admin".into();
        let message: QueuedMessage = serde_json::from_value(message).unwrap();
        assert_eq!(message.user_role, UserRole::Public);
    }

    #[tokio::test]
    async fn steering_takes_only_the_same_principal_from_the_front() {
        let registry = TaskRegistry::new(10);
        let session = "steer-session";
        let owner = queued_as("also check the tests", "test:user", UserRole::Public);
        let ctx = owner.channel_ctx.clone();
        registry.queue_message(session, owner, None).await;
        registry
            .queue_message(
                session,
                queued_as("and the docs", "test:user", UserRole::Public),
                None,
            )
            .await;
        // Another sender stops the take; what follows stays queued too.
        registry
            .queue_message(
                session,
                queued_as("hijack", "test:other", UserRole::Public),
                None,
            )
            .await;
        registry
            .queue_message(
                session,
                queued_as("later", "test:user", UserRole::Public),
                None,
            )
            .await;

        let taken = registry
            .take_steering_messages(session, UserRole::Public, &ctx)
            .await;
        let texts: Vec<&str> = taken.iter().map(|m| m.text.as_str()).collect();
        assert_eq!(texts, ["also check the tests", "and the docs"]);
        assert_eq!(registry.queue_len(session).await, 2);

        // A different role never takes the same sender's messages.
        assert!(
            registry
                .take_steering_messages(session, UserRole::Owner, &ctx)
                .await
                .is_empty()
        );
    }

    #[tokio::test]
    async fn steering_leaves_a_message_kept_for_its_own_turn_queued() {
        let registry = TaskRegistry::new(10);
        let session = "steer-separate";
        let separate = queued("what is the capital of Peru?").not_steerable();
        let ctx = separate.channel_ctx.clone();
        registry.queue_message(session, separate, None).await;
        registry
            .queue_message(session, queued("and the docs"), None)
            .await;
        assert!(
            registry
                .take_steering_messages(session, UserRole::Public, &ctx)
                .await
                .is_empty()
        );
        assert_eq!(registry.queue_len(session).await, 2);
    }

    #[tokio::test]
    async fn steering_leaves_messages_with_attachments_queued() {
        let registry = TaskRegistry::new(10);
        let session = "steer-attachments";
        let mut with_file = queued("see this file");
        with_file.attachments.push(MessageAttachment::default());
        let ctx = with_file.channel_ctx.clone();
        registry.queue_message(session, with_file, None).await;
        assert!(
            registry
                .take_steering_messages(session, UserRole::Public, &ctx)
                .await
                .is_empty()
        );
        assert_eq!(registry.queue_len(session).await, 1);
    }

    #[tokio::test]
    async fn returned_steering_messages_run_first_in_their_order() {
        let registry = TaskRegistry::new(10);
        let session = "steer-returned";
        let ctx = queued("x").channel_ctx.clone();
        registry.queue_message(session, queued("first"), None).await;
        registry
            .queue_message(session, queued("second"), None)
            .await;
        let taken = registry
            .take_steering_messages(session, UserRole::Public, &ctx)
            .await;
        assert_eq!(taken.len(), 2);
        registry
            .queue_message(session, queued("arrived later"), None)
            .await;

        registry.return_steering_messages(session, taken).await;

        let mut order = Vec::new();
        while let Some(message) = registry.pop_queued_message(session).await {
            order.push(message.text);
        }
        assert_eq!(order, ["first", "second", "arrived later"]);
    }

    #[tokio::test]
    async fn test_queue_deduplicates_identical_messages() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";

        // First message queues normally
        let result = registry
            .queue_message(session, queued("hello world"), None)
            .await;
        assert_eq!(result, Some(1));

        // Identical message within dedup window is deduplicated
        let result = registry
            .queue_message(session, queued("hello world"), None)
            .await;
        assert_eq!(result, None);

        // Different message queues normally
        let result = registry
            .queue_message(session, queued("different message"), None)
            .await;
        assert_eq!(result, Some(2));

        // Only 2 messages in queue (not 3)
        assert_eq!(registry.queue_len(session).await, 2);
    }

    #[tokio::test]
    async fn test_queue_allows_same_message_different_sessions() {
        let registry = TaskRegistry::new(10);

        let result = registry
            .queue_message("session-a", queued("hello"), None)
            .await;
        assert_eq!(result, Some(1));

        // Same message but different session — should queue
        let result = registry
            .queue_message("session-b", queued("hello"), None)
            .await;
        assert_eq!(result, Some(1));
    }

    #[tokio::test]
    async fn test_queue_deduplicates_after_pop() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";

        let result = registry.queue_message(session, queued("hello"), None).await;
        assert_eq!(result, Some(1));

        // Pop the message
        let popped = registry.pop_queued_message(session).await;
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().text, "hello");

        // Same message should STILL be deduplicated after pop (within dedup window).
        // This prevents webhook retry duplicates from re-entering the queue.
        let result = registry.queue_message(session, queued("hello"), None).await;
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_mark_seen_prevents_duplicates() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";

        // First call returns true (proceed)
        assert!(registry.mark_seen(session, "hello world", None).await);

        // Second call with same text returns false (duplicate)
        assert!(!registry.mark_seen(session, "hello world", None).await);

        // Different text returns true
        assert!(registry.mark_seen(session, "different text", None).await);

        // Different session with same text returns true
        assert!(
            registry
                .mark_seen("other-session", "hello world", None)
                .await
        );
    }

    #[tokio::test]
    async fn test_mark_seen_blocks_subsequent_queue() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";

        // mark_seen first (simulates direct processing)
        assert!(registry.mark_seen(session, "hello", None).await);

        // queue_message for the same text should be deduplicated
        let result = registry.queue_message(session, queued("hello"), None).await;
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_pop_all_coalesces_fragments() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";

        // Simulate a long message fragmented into 4 parts
        registry
            .queue_message(session, queued("Part 1: Hello"), None)
            .await;
        registry
            .queue_message(session, queued("Part 2: World"), None)
            .await;
        registry
            .queue_message(session, queued("Part 3: How are"), None)
            .await;
        registry
            .queue_message(session, queued("Part 4: you?"), None)
            .await;
        assert_eq!(registry.queue_len(session).await, 4);

        // pop_all should coalesce into a single message
        let coalesced = registry.pop_all_queued_messages(session).await;
        assert!(coalesced.is_some());
        let msg = coalesced.unwrap();
        assert_eq!(
            msg.text,
            "Part 1: Hello\nPart 2: World\nPart 3: How are\nPart 4: you?"
        );

        // Queue should now be empty
        assert_eq!(registry.queue_len(session).await, 0);
        assert!(registry.pop_all_queued_messages(session).await.is_none());
    }

    #[tokio::test]
    async fn test_pop_all_single_message() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";

        registry
            .queue_message(session, queued("only one"), None)
            .await;
        let result = registry.pop_all_queued_messages(session).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().text, "only one");
    }

    #[tokio::test]
    async fn test_queue_message_if_running_queues_while_task_runs() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";
        let (_task_id, _token) = registry.register(session, "long task").await;

        match registry
            .queue_message_if_running(session, queued("follow-up"), None)
            .await
        {
            QueueOutcome::Queued(pos) => assert_eq!(pos, 1),
            other => panic!("expected Queued(1), got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_queue_message_if_running_rejects_when_no_task() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";

        // No running task: the caller must process directly instead of
        // queueing into a queue nobody will ever drain.
        assert!(matches!(
            registry
                .queue_message_if_running(session, queued("orphan"), None)
                .await,
            QueueOutcome::NoRunningTask
        ));
        assert_eq!(registry.queue_len(session).await, 0);
    }

    #[tokio::test]
    async fn test_queue_message_if_running_detects_finished_task_race() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";
        let (task_id, _token) = registry.register(session, "task").await;
        // Task finishes between the caller's has_running_task check and the
        // queue attempt — the registry must report NoRunningTask instead of
        // stranding the message.
        registry.complete(task_id).await;

        assert!(matches!(
            registry
                .queue_message_if_running(session, queued("late message"), None)
                .await,
            QueueOutcome::NoRunningTask
        ));
    }

    #[tokio::test]
    async fn test_queue_message_if_running_deduplicates() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";
        let (_task_id, _token) = registry.register(session, "task").await;

        assert!(matches!(
            registry
                .queue_message_if_running(session, queued("hello"), None)
                .await,
            QueueOutcome::Queued(1)
        ));
        assert!(matches!(
            registry
                .queue_message_if_running(session, queued("hello"), None)
                .await,
            QueueOutcome::Duplicate
        ));
    }

    #[tokio::test]
    async fn test_finalize_and_drain_completes_and_returns_queued_work() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";
        let (task_id, _token) = registry.register(session, "task").await;
        registry
            .queue_message_if_running(session, queued("queued A"), None)
            .await;
        registry
            .queue_message_if_running(session, queued("queued B"), None)
            .await;

        let drained = registry.finalize_and_drain(task_id, session, None).await;
        assert_eq!(drained.unwrap().text, "queued A\nqueued B");
        assert!(!registry.has_running_task(session).await);
        assert_eq!(registry.queue_len(session).await, 0);
    }

    #[tokio::test]
    async fn queued_group_messages_never_inherit_another_senders_role_or_attachments() {
        let registry = TaskRegistry::new(10);
        let session = "shared-group";
        let (initial_task_id, _token) = registry.register(session, "initial turn").await;

        let mut owner = queued_as("owner request", "slack:owner", UserRole::Owner);
        owner.attachments.push(MessageAttachment {
            filename: "owner.txt".to_string(),
            ..MessageAttachment::default()
        });
        let mut guest_a = queued_as("guest part A", "slack:guest", UserRole::Guest);
        guest_a.attachments.push(MessageAttachment {
            filename: "guest-a.txt".to_string(),
            ..MessageAttachment::default()
        });
        let mut guest_b = queued_as("guest part B", "slack:guest", UserRole::Guest);
        guest_b.attachments.push(MessageAttachment {
            filename: "guest-b.txt".to_string(),
            ..MessageAttachment::default()
        });

        registry
            .queue_message_if_running(session, owner, Some("owner-message"))
            .await;
        registry
            .queue_message_if_running(session, guest_a, Some("guest-message-a"))
            .await;
        registry
            .queue_message_if_running(session, guest_b, Some("guest-message-b"))
            .await;

        let owner_turn = registry
            .finalize_and_drain(initial_task_id, session, None)
            .await
            .expect("owner turn");
        assert_eq!(owner_turn.user_role, UserRole::Owner);
        assert_eq!(
            owner_turn.channel_ctx.sender_id.as_deref(),
            Some("slack:owner")
        );
        assert_eq!(owner_turn.text, "owner request");
        assert_eq!(owner_turn.attachments[0].filename, "owner.txt");
        assert_eq!(registry.queue_len(session).await, 2);

        let (owner_task_id, _token) = registry.register(session, "owner turn").await;
        let guest_turn = registry
            .finalize_and_drain(owner_task_id, session, None)
            .await
            .expect("guest turn");
        assert_eq!(guest_turn.user_role, UserRole::Guest);
        assert_eq!(
            guest_turn.channel_ctx.sender_id.as_deref(),
            Some("slack:guest")
        );
        assert_eq!(guest_turn.text, "guest part A\nguest part B");
        assert_eq!(
            guest_turn
                .attachments
                .iter()
                .map(|attachment| attachment.filename.as_str())
                .collect::<Vec<_>>(),
            vec!["guest-a.txt", "guest-b.txt"]
        );
        assert_eq!(registry.queue_len(session).await, 0);
    }

    #[tokio::test]
    async fn test_finalize_and_drain_with_error_marks_failed() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";
        let (task_id, _token) = registry.register(session, "task").await;

        let drained = registry
            .finalize_and_drain(task_id, session, Some("boom"))
            .await;
        assert!(drained.is_none());
        assert!(!registry.has_running_task(session).await);
        let entries = registry.list_for_session(session).await;
        assert!(matches!(entries[0].status, TaskStatus::Failed(_)));
    }

    #[tokio::test]
    async fn cancellation_preserves_terminal_cause_and_hands_off_fifo() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";
        let (task_id, _token) = registry.register(session, "task").await;
        registry
            .queue_message_if_running(session, queued("next"), Some("next-1"))
            .await;

        assert!(registry.cancel(task_id).await);
        let drained = registry
            .finalize_and_drain(task_id, session, Some("Task cancelled"))
            .await
            .expect("queued work must be handed off after cancellation");

        assert_eq!(drained.text, "next");
        assert_eq!(registry.queue_len(session).await, 0);
        let entries = registry.list_for_session(session).await;
        assert!(matches!(entries[0].status, TaskStatus::Cancelled));
    }

    #[tokio::test]
    async fn cancellation_request_keeps_fifo_owner_active_until_handoff() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";
        let (task_id, _token) = registry.register(session, "task").await;
        assert!(registry.cancel(task_id).await);
        assert!(matches!(
            registry.list_for_session(session).await[0].status,
            TaskStatus::Cancelling
        ));

        let outcome = registry
            .queue_message_if_running(session, queued("arrived during cancellation"), Some("m-2"))
            .await;
        assert!(matches!(outcome, QueueOutcome::Queued(1)));
        let drained = registry
            .finalize_and_drain(task_id, session, Some("Task cancelled"))
            .await
            .expect("cancelling worker retains FIFO ownership");
        assert_eq!(drained.text, "arrived during cancellation");
    }

    // Regression: a user re-typing the SAME text gets a new per-message
    // identity (Telegram message_id / Slack ts / Discord id). With a distinct
    // dedup_key, the re-ask must be treated as a fresh request — not silently
    // swallowed by the 120s text-dedup window. (Screenshot bug: "who is my
    // wife?" sent twice, second got no reply.)
    #[tokio::test]
    async fn test_mark_seen_allows_reask_with_distinct_message_id() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";

        // First send of "who is my wife?" with message_id 100.
        assert!(
            registry
                .mark_seen(session, "who is my wife?", Some("100"))
                .await
        );

        // Deliberate re-ask: identical text, NEW message_id 101 → proceed.
        assert!(
            registry
                .mark_seen(session, "who is my wife?", Some("101"))
                .await
        );

        // True redelivery of message_id 100 (poll/webhook retry) → dropped.
        assert!(
            !registry
                .mark_seen(session, "who is my wife?", Some("100"))
                .await
        );
    }

    // Regression: the queued path (task already running) must also key on the
    // per-message identity so a re-ask queues behind the running task instead
    // of being dropped as a Duplicate.
    #[tokio::test]
    async fn test_queue_if_running_allows_reask_with_distinct_message_id() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";
        let (_task_id, _token) = registry.register(session, "task").await;

        assert!(matches!(
            registry
                .queue_message_if_running(session, queued("status?"), Some("200"))
                .await,
            QueueOutcome::Queued(1)
        ));
        // Same text, new message_id → fresh request, queues again.
        assert!(matches!(
            registry
                .queue_message_if_running(session, queued("status?"), Some("201"))
                .await,
            QueueOutcome::Queued(2)
        ));
        // Redelivery of message_id 200 → dropped as Duplicate.
        assert!(matches!(
            registry
                .queue_message_if_running(session, queued("status?"), Some("200"))
                .await,
            QueueOutcome::Duplicate
        ));
    }

    #[tokio::test]
    async fn test_stale_running_task_is_reaped_and_unblocks_queue() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";
        let (task_id, _token) = registry.register(session, "leaked task").await;

        // Force the task to look leaked: started long before the staleness cap
        // and never finalized (simulates an abnormal channel-handler exit).
        {
            let mut tasks = registry.tasks.write().await;
            let handle = tasks.get_mut(&task_id).unwrap();
            handle.entry.started_at =
                Utc::now() - chrono::Duration::seconds(MAX_RUNNING_TASK_AGE_SECS + 60);
        }

        // The phantom must not report as running...
        assert!(!registry.has_running_task(session).await);
        // ...and a new message must process directly, not queue behind it.
        assert!(matches!(
            registry
                .queue_message_if_running(session, queued("new message"), Some("m1"))
                .await,
            QueueOutcome::NoRunningTask
        ));
        // The reaped task is surfaced as Failed (resolved) in listings.
        let entries = registry.list_for_session(session).await;
        assert!(matches!(entries[0].status, TaskStatus::Failed(_)));
    }

    #[tokio::test]
    async fn test_recent_running_task_is_not_reaped() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";
        let (_task_id, _token) = registry.register(session, "active task").await;

        // A freshly-registered task is genuinely running and must still gate
        // the queue (no false-positive reaping of legitimate work).
        assert!(registry.has_running_task(session).await);
        assert!(matches!(
            registry
                .queue_message_if_running(session, queued("follow-up"), Some("m1"))
                .await,
            QueueOutcome::Queued(1)
        ));
    }

    #[tokio::test]
    async fn test_finalize_and_drain_empty_queue_returns_none() {
        let registry = TaskRegistry::new(10);
        let session = "test-session";
        let (task_id, _token) = registry.register(session, "task").await;

        assert!(
            registry
                .finalize_and_drain(task_id, session, None)
                .await
                .is_none()
        );
        assert!(!registry.has_running_task(session).await);
    }
}

#[cfg(test)]
mod queue_edit_tests {
    use super::*;

    #[tokio::test]
    async fn queued_messages_list_edit_and_remove_by_id() {
        let registry = TaskRegistry::new(10);
        let session = "app:synthetic-q";
        let ctx = ChannelContext::private("app");
        let first = QueuedMessage::new("first", UserRole::Owner, ctx.clone(), Vec::new());
        let second = QueuedMessage::new("second", UserRole::Owner, ctx, Vec::new());
        let (a, b) = (first.id, second.id);
        registry.queue_message(session, first, None).await;
        registry.queue_message(session, second, None).await;
        assert!(
            registry
                .edit_queued_message(session, a, Some("first, edited".into()))
                .await
        );
        assert!(registry.edit_queued_message(session, b, None).await);
        assert!(!registry.edit_queued_message(session, b, None).await);
        let left: Vec<String> = registry
            .queued_messages(session)
            .await
            .into_iter()
            .map(|(_, text, _, _)| text)
            .collect();
        assert_eq!(left, ["first, edited"]);
    }

    #[tokio::test]
    async fn a_message_for_the_running_work_is_taken_past_one_queued_for_later() {
        let registry = TaskRegistry::new(10);
        let session = "app:synthetic-steer";
        let ctx = ChannelContext::private("app");
        let later = QueuedMessage::new("after this", UserRole::Owner, ctx.clone(), Vec::new())
            .placed_for_later();
        let now = QueuedMessage::new("also this", UserRole::Owner, ctx.clone(), Vec::new());
        registry.queue_message(session, later, None).await;
        registry.queue_message(session, now, None).await;
        let taken = registry
            .take_steering_messages(session, UserRole::Owner, &ctx)
            .await;
        assert_eq!(
            taken.iter().map(|m| m.text.as_str()).collect::<Vec<_>>(),
            ["also this"]
        );
        let left: Vec<String> = registry
            .queued_messages(session)
            .await
            .into_iter()
            .map(|(_, text, _, _)| text)
            .collect();
        assert_eq!(left, ["after this"]);
    }

    /// A device's ceiling does not make its message another sender's: a
    /// running turn takes it (and takes the ceiling on), while a message
    /// from another conversation still waits for its own turn.
    #[tokio::test]
    async fn a_capped_devices_steer_is_the_same_sender() {
        let registry = TaskRegistry::new(10);
        let session = "app:synthetic-capped-steer";
        let ctx = ChannelContext::private("app");
        let mut capped = ctx.clone();
        capped.sender_ceiling = Some(crate::context::SenderCeiling::Medium);
        let mut elsewhere = ctx.clone();
        elsewhere.channel_id = Some("app:synthetic-other".to_string());
        for (text, from) in [
            ("from the reader", capped.clone()),
            ("from the phone", ctx.clone()),
            ("from elsewhere", elsewhere),
        ] {
            let message = QueuedMessage::new(text, UserRole::Owner, from, Vec::new());
            registry.queue_message(session, message, None).await;
        }
        let taken = registry
            .take_steering_messages(session, UserRole::Owner, &ctx)
            .await;
        assert_eq!(
            taken.iter().map(|m| m.text.as_str()).collect::<Vec<_>>(),
            ["from the reader", "from the phone"]
        );
        assert_eq!(
            taken[0].channel_ctx.sender_ceiling,
            Some(crate::context::SenderCeiling::Medium),
            "the ceiling travels with the message"
        );
        assert_eq!(registry.queue_len(session).await, 1);
        // A capped turn takes the owner's own steer too.
        let phone = QueuedMessage::new("one more", UserRole::Owner, ctx.clone(), Vec::new());
        let session = "app:synthetic-capped-turn";
        registry.queue_message(session, phone, None).await;
        assert_eq!(
            registry
                .take_steering_messages(session, UserRole::Owner, &capped)
                .await
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn messages_queued_for_later_each_run_as_their_own_turn() {
        let registry = TaskRegistry::new(10);
        let session = "app:synthetic-separate-turns";
        let ctx = ChannelContext::private("app");
        for text in ["draft the invoice", "then plan the trip"] {
            let later = QueuedMessage::new(text, UserRole::Owner, ctx.clone(), Vec::new())
                .placed_for_later();
            registry.queue_message(session, later, None).await;
        }
        let first = registry.pop_all_queued_messages(session).await.unwrap();
        assert_eq!(first.text, "draft the invoice");
        let second = registry.pop_all_queued_messages(session).await.unwrap();
        assert_eq!(second.text, "then plan the trip");
    }

    #[tokio::test]
    async fn a_leftover_message_for_the_work_runs_before_one_queued_for_later() {
        let registry = TaskRegistry::new(10);
        let session = "app:synthetic-drain";
        let ctx = ChannelContext::private("app");
        let later = QueuedMessage::new("after this", UserRole::Owner, ctx.clone(), Vec::new())
            .placed_for_later();
        let now = QueuedMessage::new("also this", UserRole::Owner, ctx, Vec::new());
        registry.queue_message(session, later, None).await;
        registry.queue_message(session, now, None).await;
        let first = registry.pop_all_queued_messages(session).await.unwrap();
        assert_eq!(first.text, "also this");
        let second = registry.pop_all_queued_messages(session).await.unwrap();
        assert_eq!(second.text, "after this");
    }

    /// "Send now" on a message waiting for its own turn: the running work
    /// takes it at its next step, as it takes one sent to be read now. A
    /// message that was not sent to wait (or is gone) is left as it is.
    #[tokio::test]
    async fn a_message_queued_for_later_can_be_given_to_the_running_work() {
        let registry = TaskRegistry::new(10);
        let session = "app:synthetic-send-now";
        let ctx = ChannelContext::private("app");
        let later = QueuedMessage::new("after this", UserRole::Owner, ctx.clone(), Vec::new())
            .placed_for_later();
        let later_id = later.id;
        let unrelated =
            QueuedMessage::new("another task", UserRole::Owner, ctx.clone(), Vec::new())
                .not_steerable();
        let unrelated_id = unrelated.id;
        registry.queue_message(session, later, None).await;
        registry.queue_message(session, unrelated, None).await;
        assert!(
            registry
                .take_steering_messages(session, UserRole::Owner, &ctx)
                .await
                .is_empty()
        );

        assert!(registry.send_queued_now(session, later_id).await);
        assert!(!registry.send_queued_now(session, unrelated_id).await);
        assert!(!registry.send_queued_now(session, 0).await);
        let taken = registry
            .take_steering_messages(session, UserRole::Owner, &ctx)
            .await;
        assert_eq!(
            taken.iter().map(|m| m.text.as_str()).collect::<Vec<_>>(),
            ["after this"]
        );
        assert!(
            !registry.send_queued_now(session, later_id).await,
            "already taken"
        );
    }
}
