//! Everything a chat has running, in one list, and stopping it.
//!
//! The list joins three typed sources: the chat's turns (the
//! [`TaskRegistry`]), and its sub-agents and background commands (the
//! agent's [`ChatWorkView`]). `/tasks`, `/status`, `/cancel` and a stop
//! request read in plain words all go through it, and every stop is read
//! back before it is reported: a turn counts as stopped once the registry
//! no longer runs it, a sub-agent or command once the agent says it is gone.

use std::time::Duration;

use aidaemon_tools::duration_format::{compact_chrono_duration, ZeroUnitStyle};
use chrono::{DateTime, Utc};

use crate::ports::{ChannelAgentRuntime, ChatWorkView, RunningItem};
use crate::tasks::{TaskRegistry, TaskStatus};

/// How `/status`, `/tasks` and stop reports name a chat turn.
pub const TURN_LABEL: &str = "Working on your message";

/// How long a stopped turn gets to end before the report says it has not.
const TURN_SETTLE: Duration = Duration::from_secs(5);
const TURN_POLL: Duration = Duration::from_millis(100);

/// What one entry of the list stops.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Target {
    Turn(u64),
    Work(String),
}

/// One thing running for the chat.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkEntry {
    /// The short number `/cancel <n>` takes.
    pub number: u64,
    /// What it is, in plain words.
    pub title: String,
    pub started_at: DateTime<Utc>,
    /// The request that started it: entries sharing one are stopped together.
    group: String,
    target: Target,
}

/// Everything running for a chat, oldest first.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WorkList {
    pub entries: Vec<WorkEntry>,
}

impl WorkList {
    /// How many requests the running entries come from.
    pub fn requests(&self) -> usize {
        let mut groups: Vec<&str> = self.entries.iter().map(|e| e.group.as_str()).collect();
        groups.sort_unstable();
        groups.dedup();
        groups.len()
    }

    /// The entries as the mid-run reader sees them.
    pub fn items(&self, now: DateTime<Utc>) -> Vec<RunningItem> {
        self.entries
            .iter()
            .map(|entry| RunningItem {
                number: entry.number,
                title: entry.title.clone(),
                elapsed: (now - entry.started_at).to_std().unwrap_or_default(),
            })
            .collect()
    }

    /// One line per entry: its number, title and how long it has run.
    pub fn lines(&self, now: DateTime<Utc>) -> Vec<String> {
        self.entries
            .iter()
            .map(|entry| {
                format!(
                    "#{} {} ({} so far)",
                    entry.number,
                    crate::presentation::word_preview(&entry.title, 120),
                    compact_chrono_duration(now - entry.started_at, ZeroUnitStyle::Trim)
                )
            })
            .collect()
    }
}

#[cfg(test)]
impl WorkList {
    /// A list of `(number, title, started_at, request)` entries.
    pub(crate) fn for_tests(entries: Vec<(u64, &str, DateTime<Utc>, &str)>) -> Self {
        let mut entries: Vec<WorkEntry> = entries
            .into_iter()
            .map(|(number, title, started_at, group)| WorkEntry {
                number,
                title: title.to_string(),
                started_at,
                group: group.to_string(),
                target: Target::Work(format!("test:{number}")),
            })
            .collect();
        entries.sort_by_key(|entry| entry.started_at);
        Self { entries }
    }
}

/// The chat's running turns, sub-agents and background commands.
pub async fn running_work(
    registry: &TaskRegistry,
    agent: &dyn ChannelAgentRuntime,
    session_id: &str,
) -> WorkList {
    let view = agent.chat_work(session_id).await;
    build_list(registry, session_id, view).await
}

async fn build_list(registry: &TaskRegistry, session_id: &str, view: ChatWorkView) -> WorkList {
    let mut entries: Vec<WorkEntry> = registry
        .list_for_session(session_id)
        .await
        .into_iter()
        .filter(|entry| matches!(entry.status, TaskStatus::Running))
        .map(|entry| WorkEntry {
            number: entry.id,
            title: TURN_LABEL.to_string(),
            started_at: entry.started_at,
            group: view
                .current_request
                .clone()
                .unwrap_or_else(|| format!("turn:{}", entry.id)),
            target: Target::Turn(entry.id),
        })
        .collect();
    let keys: Vec<&str> = view.items.iter().map(|item| item.key.as_str()).collect();
    let numbers = registry.number_work(session_id, &keys);
    for (item, number) in view.items.iter().zip(numbers) {
        entries.push(WorkEntry {
            number,
            title: item.title.clone(),
            started_at: item.started_at,
            group: item
                .request
                .clone()
                .unwrap_or_else(|| format!("work:{}", item.key)),
            target: Target::Work(item.key.clone()),
        });
    }
    entries.sort_by_key(|entry| entry.started_at);
    WorkList { entries }
}

/// What a stop asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopRequest {
    /// Bare `/cancel` (or a stop word): everything, when it all comes from
    /// one request; otherwise the list, to pick from.
    Bare,
    /// `/cancel all`: everything.
    All,
    /// These numbers.
    Numbers(Vec<u64>),
}

/// Stop what `request` names and report, in plain words, what was stopped
/// as read back afterwards.
pub async fn stop(
    registry: &TaskRegistry,
    agent: &dyn ChannelAgentRuntime,
    session_id: &str,
    request: StopRequest,
) -> String {
    let list = running_work(registry, agent, session_id).await;
    let now = Utc::now();
    let (chosen, everything): (Vec<WorkEntry>, bool) = match &request {
        StopRequest::Numbers(numbers) => {
            let chosen: Vec<WorkEntry> = list
                .entries
                .iter()
                .filter(|entry| numbers.contains(&entry.number))
                .cloned()
                .collect();
            if chosen.is_empty() {
                let named = numbers
                    .iter()
                    .map(|n| format!("#{n}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                return format!(
                    "Nothing running has the number {named}. Send /tasks to see what's running."
                );
            }
            (chosen, false)
        }
        StopRequest::Bare if list.requests() > 1 => {
            return format!(
                "Work from more than one request is running:\n{}\n\nSend /cancel <number> to \
                 stop one, or /cancel all to stop everything.",
                list.lines(now).join("\n")
            );
        }
        StopRequest::Bare | StopRequest::All => (list.entries.clone(), true),
    };

    let goals = if everything {
        agent.cancel_finite_goals_for_session(session_id).await
    } else {
        Vec::new()
    };
    if chosen.is_empty() && goals.is_empty() {
        return "Nothing is running in this chat right now.".to_string();
    }

    // Sub-agents and commands first: stopping the turn first would drop its
    // waited-on sub-agents before their commands could be stopped.
    let (turns, work): (Vec<&WorkEntry>, Vec<&WorkEntry>) = chosen
        .iter()
        .partition(|entry| matches!(entry.target, Target::Turn(_)));
    let mut work_results = Vec::with_capacity(work.len());
    for entry in work {
        if let Target::Work(key) = &entry.target {
            work_results.push((
                entry.title.clone(),
                agent.stop_chat_work(session_id, key).await,
            ));
        }
    }
    let mut stopped: Vec<String> = Vec::new();
    let mut still: Vec<String> = Vec::new();
    for entry in turns {
        let Target::Turn(id) = entry.target else {
            continue;
        };
        registry.cancel(id).await;
        if turn_ended(registry, id).await {
            stopped.push(entry.title.clone());
        } else {
            still.push(entry.title.clone());
        }
    }
    for (title, ok) in work_results {
        if ok {
            stopped.push(title);
        } else {
            still.push(title);
        }
    }
    stopped.extend(goals);

    let mut reply = String::new();
    if !stopped.is_empty() {
        reply.push_str("⏹️ Stopped:");
        for title in &stopped {
            reply.push_str(&format!(
                "\n• {}",
                crate::presentation::word_preview(title, 120)
            ));
        }
    }
    if !still.is_empty() {
        if !reply.is_empty() {
            reply.push_str("\n\n");
        }
        reply.push_str("I asked these to stop, but they were still running when I checked:");
        for title in &still {
            reply.push_str(&format!(
                "\n• {}",
                crate::presentation::word_preview(title, 120)
            ));
        }
        reply.push_str("\nSend /tasks to check again.");
    }
    let queued = registry.queue_len(session_id).await;
    if queued > 0 && stopped.iter().any(|title| title == TURN_LABEL) {
        reply.push_str(&format!(
            "\n\nYour {queued} waiting message{} will still run in order.",
            if queued == 1 { "" } else { "s" }
        ));
    }
    reply
}

/// Whether turn `id` has ended (or is no longer kept), reading the registry
/// until it has or [`TURN_SETTLE`] passes.
async fn turn_ended(registry: &TaskRegistry, id: u64) -> bool {
    let deadline = tokio::time::Instant::now() + TURN_SETTLE;
    loop {
        let active = registry.entry(id).await.is_some_and(|entry| {
            matches!(entry.status, TaskStatus::Running | TaskStatus::Cancelling)
        });
        if !active {
            return true;
        }
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(TURN_POLL).await;
    }
}
