use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug)]
pub enum TaskStatus {
    Running,
    Completed,
    Failed(String),
    Cancelled,
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskStatus::Running => write!(f, "Running"),
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
}

pub struct TaskRegistry {
    tasks: RwLock<HashMap<u64, TaskHandle>>,
    next_id: AtomicU64,
    max_completed: usize,
}

impl TaskRegistry {
    pub fn new(max_completed: usize) -> Self {
        Self {
            tasks: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            max_completed,
        }
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
        };
        let mut tasks = self.tasks.write().await;
        tasks.insert(id, handle);
        (id, cancel_token)
    }

    /// Mark a task as completed.
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

    /// Cancel a running task. Returns true if the task was found and cancelled.
    pub async fn cancel(&self, task_id: u64) -> bool {
        let mut tasks = self.tasks.write().await;
        if let Some(handle) = tasks.get_mut(&task_id) {
            if matches!(handle.entry.status, TaskStatus::Running) {
                handle.cancel_token.cancel();
                handle.entry.status = TaskStatus::Cancelled;
                handle.entry.finished_at = Some(Utc::now());
                return true;
            }
        }
        false
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

    /// Remove oldest finished tasks when count exceeds max_completed.
    fn cleanup_locked(tasks: &mut HashMap<u64, TaskHandle>, max_completed: usize) {
        let mut finished: Vec<u64> = tasks
            .iter()
            .filter(|(_, h)| !matches!(h.entry.status, TaskStatus::Running))
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
}
