//! Conflict-aware local workspace checkpoints backed by an external bare Git
//! object store.
//!
//! Checkpoint commands never run Git porcelain against the user's repository:
//! every Git process receives an explicit external `GIT_DIR`, a temporary
//! external index, and empty system/global configuration.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use ignore::WalkBuilder;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha1::{Digest, Sha1};
use sha2::Sha256;
use sqlx::{Row, SqlitePool};
use tokio::sync::{Mutex, OwnedMutexGuard};
use tracing::{info, warn};
use uuid::Uuid;

use crate::config::CheckpointConfig;
use crate::events::{Event, EventStore, EventType};
use crate::execution::{BackendKind, SharedExecutionBackend};
use crate::tools::fs_utils;

static ACTIVE_MANAGER: Lazy<RwLock<Option<Arc<CheckpointManager>>>> =
    Lazy::new(|| RwLock::new(None));

const READY: &str = "ready";
const OPEN: &str = "open";
const UNSAFE: &str = "unsafe";
const ROLLBACK_CONFIRMATION_MINUTES: i64 = 5;

pub(crate) fn active_manager() -> Option<Arc<CheckpointManager>> {
    ACTIVE_MANAGER
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// Install the process-wide manager after the immutable execution backend is
/// selected. Remote backends are deliberately unsupported in the beta.
pub(crate) async fn install_manager(
    config: &CheckpointConfig,
    pool: SqlitePool,
    event_store: Arc<EventStore>,
    backend: SharedExecutionBackend,
) -> anyhow::Result<Option<Arc<CheckpointManager>>> {
    if !config.enabled {
        return Ok(None);
    }
    if backend.kind() != BackendKind::Local {
        warn!(
            backend = backend.kind().as_str(),
            "Filesystem checkpoints are local-only; checkpointing is disabled for this backend"
        );
        return Ok(None);
    }

    if let Some(existing) = active_manager() {
        return Ok(Some(existing));
    }

    let manager =
        Arc::new(CheckpointManager::new(config.clone(), pool, event_store, backend).await?);
    manager.reconcile_interrupted_state().await?;
    manager.prune_retention().await?;

    let mut slot = ACTIVE_MANAGER
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(existing) = slot.as_ref() {
        return Ok(Some(existing.clone()));
    }
    *slot = Some(manager.clone());
    info!(
        store = %manager.store_root.display(),
        "Local filesystem checkpoints enabled"
    );
    Ok(Some(manager))
}

#[derive(Debug, Clone)]
struct SnapshotStats {
    tree: String,
    included_paths: usize,
    included_bytes: u64,
    excluded_paths: usize,
}

#[derive(Debug, Clone)]
struct SnapshotFile {
    relative: String,
    mode: u32,
}

#[derive(Debug, Clone)]
struct CheckpointRecord {
    id: String,
    root: PathBuf,
    repo: PathBuf,
    pre_tree: String,
    post_tree: Option<String>,
    state: String,
    included_paths: usize,
    included_bytes: u64,
    unsafe_reason: Option<String>,
    rollback_of: Option<String>,
    created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct TreeEntry {
    mode: u32,
    oid: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RestoreOp {
    path: String,
    expected_current: Option<TreeEntry>,
    restore: Option<TreeEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RestoreConflict {
    path: String,
    reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RollbackPlan {
    checkpoint_id: String,
    root: PathBuf,
    repo: PathBuf,
    pre_tree: String,
    post_tree: String,
    operations: Vec<RestoreOp>,
    conflicts: Vec<RestoreConflict>,
}

#[derive(Debug, Clone)]
struct Confirmation {
    session_id: String,
    expires_at: DateTime<Utc>,
    plan: RollbackPlan,
}

#[derive(Debug, Clone)]
pub(crate) struct RollbackPreview {
    pub checkpoint_id: String,
    pub root: PathBuf,
    pub writes: usize,
    pub deletes: usize,
    pub conflicts: usize,
    pub samples: Vec<String>,
    pub token: String,
    pub expires_at: DateTime<Utc>,
}

impl RollbackPreview {
    pub(crate) fn render(&self) -> String {
        let samples = if self.samples.is_empty() {
            String::new()
        } else {
            format!("\nPaths:\n{}", self.samples.join("\n"))
        };
        format!(
            "Rollback preview for {}:\nRoot: {}\nRestore/write: {}\nDelete: {}\n\
             Preserved conflicts: {}{}\n\nNo files have changed. To apply this exact preview, \
             send:\n/rollback confirm {}\n\nConfirmation expires at {}.",
            self.checkpoint_id,
            self.root.display(),
            self.writes,
            self.deletes,
            self.conflicts,
            samples,
            self.token,
            self.expires_at.to_rfc3339()
        )
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RollbackResult {
    checkpoint_id: String,
    safety_checkpoint_id: String,
    applied: usize,
    conflicts: Vec<RestoreConflict>,
}

impl RollbackResult {
    pub(crate) fn render(&self) -> String {
        let conflict_text = if self.conflicts.is_empty() {
            String::new()
        } else {
            let samples = self
                .conflicts
                .iter()
                .take(12)
                .map(|conflict| format!("• {} — {}", conflict.path, conflict.reason))
                .collect::<Vec<_>>()
                .join("\n");
            format!("\nPreserved conflicts:\n{}", samples)
        };
        format!(
            "Rollback {} completed: {} path(s) restored; {} conflict(s) were left untouched.\n\
             Safety checkpoint: {}{}",
            self.checkpoint_id,
            self.applied,
            self.conflicts.len(),
            self.safety_checkpoint_id,
            conflict_text
        )
    }
}

pub(crate) struct CheckpointManager {
    config: CheckpointConfig,
    pool: SqlitePool,
    event_store: Arc<EventStore>,
    backend: SharedExecutionBackend,
    store_root: PathBuf,
    git_binary: PathBuf,
    git_config: PathBuf,
    root_locks: Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>,
    leases: Mutex<HashMap<String, OwnedMutexGuard<()>>>,
    confirmations: Mutex<HashMap<String, Confirmation>>,
}

impl CheckpointManager {
    async fn new(
        config: CheckpointConfig,
        pool: SqlitePool,
        event_store: Arc<EventStore>,
        backend: SharedExecutionBackend,
    ) -> anyhow::Result<Self> {
        validate_config(&config)?;
        let store_root = checkpoint_store_path(&config)?;
        validate_store_path(&store_root)?;
        fs::create_dir_all(&store_root)?;
        let store_root = store_root.canonicalize()?;

        let workspace = PathBuf::from(backend.workspace_root().as_str());
        if let Ok(workspace) = workspace.canonicalize() {
            anyhow::ensure!(
                !store_root.starts_with(&workspace),
                "checkpoint storage must be outside the execution workspace ({})",
                workspace.display()
            );
        }
        validate_store_path(&store_root)?;
        set_owner_only(&store_root)?;

        let git_binary = discover_git()?;
        let git_config = store_root.join("empty-gitconfig");
        if !git_config.exists() {
            File::create(&git_config)?;
        }
        set_owner_only_file(&git_config)?;
        let output = isolated_git_command(&git_binary, &git_config, None)
            .arg("--version")
            .output()?;
        ensure_success("git --version", &output)?;

        Ok(Self {
            config,
            pool,
            event_store,
            backend,
            store_root,
            git_binary,
            git_config,
            root_locks: Mutex::new(HashMap::new()),
            leases: Mutex::new(HashMap::new()),
            confirmations: Mutex::new(HashMap::new()),
        })
    }

    pub(crate) async fn begin_for_tool(
        &self,
        tool: &str,
        arguments: &str,
    ) -> anyhow::Result<Option<String>> {
        if !self.config.enabled || !tool_action_can_mutate(tool, arguments) {
            return Ok(None);
        }

        let args = serde_json::from_str::<Value>(arguments).unwrap_or(Value::Null);
        let session_id = internal_string(&args, "_session_id").unwrap_or_else(|| "unknown".into());
        let task_id = internal_string(&args, "_task_id");
        let turn_id = internal_string(&args, "_turn_id");
        let scope_id = if let Some(turn_id) = turn_id.as_deref() {
            format!("turn:{turn_id}")
        } else if let Some(task_id) = task_id.as_deref() {
            format!("task:{task_id}")
        } else {
            format!("call:{}", Uuid::new_v4())
        };

        let root = match self.resolve_root(tool, &args).await {
            Ok(root) => root,
            Err(error) => {
                self.emit(
                    &session_id,
                    EventType::CheckpointSkipped,
                    json!({
                        "task_id": task_id,
                        "turn_id": turn_id,
                        "name": tool,
                        "reason": error.to_string(),
                    }),
                )
                .await;
                return Err(error);
            }
        };
        let backend_id = self.backend.id().to_string();

        if let Some(id) = self
            .existing_checkpoint_id(&scope_id, &backend_id, &root)
            .await?
        {
            return Ok(Some(id));
        }

        self.prune_retention().await?;
        let size = directory_size(&self.store_root)?;
        anyhow::ensure!(
            size < self.config.max_store_bytes,
            "checkpoint store is at its {} byte limit; prune checkpoints before mutating",
            self.config.max_store_bytes
        );

        let root_lock = self.root_lock(&root).await;
        let guard = root_lock.lock_owned().await;
        if let Some(id) = self
            .existing_checkpoint_id(&scope_id, &backend_id, &root)
            .await?
        {
            drop(guard);
            return Ok(Some(id));
        }

        let id = checkpoint_id("cp");
        let repo = self.ensure_root_repo(&root)?;
        let snapshot = match self
            .snapshot_with_timeout(root.clone(), repo.clone(), id.clone(), "pre")
            .await
        {
            Ok(snapshot) => snapshot,
            Err(error) => {
                self.emit(
                    &session_id,
                    EventType::CheckpointSkipped,
                    json!({
                        "checkpoint_id": id,
                        "task_id": task_id,
                        "turn_id": turn_id,
                        "name": tool,
                        "root": root,
                        "reason": error.to_string(),
                    }),
                )
                .await;
                return Err(error);
            }
        };
        let now = Utc::now();
        let expires = now + chrono::Duration::days(self.config.retention_days as i64);
        sqlx::query(
            r#"
            INSERT INTO filesystem_checkpoints
                (id, scope_id, session_id, task_id, turn_id, backend_id, root_path,
                 store_path, pre_tree, state, origin_tool, included_paths,
                 included_bytes, excluded_paths, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&id)
        .bind(&scope_id)
        .bind(&session_id)
        .bind(task_id.as_deref())
        .bind(turn_id.as_deref())
        .bind(&backend_id)
        .bind(root.to_string_lossy().as_ref())
        .bind(repo.to_string_lossy().as_ref())
        .bind(&snapshot.tree)
        .bind(OPEN)
        .bind(tool)
        .bind(snapshot.included_paths as i64)
        .bind(snapshot.included_bytes as i64)
        .bind(snapshot.excluded_paths as i64)
        .bind(now.to_rfc3339())
        .bind(expires.to_rfc3339())
        .execute(&self.pool)
        .await?;
        self.leases.lock().await.insert(id.clone(), guard);
        self.emit(
            &session_id,
            EventType::CheckpointCreated,
            json!({
                "checkpoint_id": id,
                "task_id": task_id,
                "turn_id": turn_id,
                "name": tool,
                "root": root,
                "tree": snapshot.tree,
                "included_paths": snapshot.included_paths,
                "included_bytes": snapshot.included_bytes,
                "excluded_paths": snapshot.excluded_paths,
            }),
        )
        .await;
        Ok(Some(id))
    }

    pub(crate) async fn mark_task_unsafe(&self, task_id: &str, reason: &str) {
        if let Err(error) = sqlx::query(
            "UPDATE filesystem_checkpoints
             SET unsafe_reason = COALESCE(unsafe_reason, ?)
             WHERE task_id = ? AND state = ?",
        )
        .bind(reason)
        .bind(task_id)
        .bind(OPEN)
        .execute(&self.pool)
        .await
        {
            warn!(task_id, error = %error, "Failed to mark checkpoint unsafe");
        }
    }

    pub(crate) async fn finalize_task(
        &self,
        task_id: &str,
        session_id: &str,
    ) -> anyhow::Result<()> {
        let rows = sqlx::query(
            "SELECT id, root_path, store_path, unsafe_reason, turn_id
             FROM filesystem_checkpoints WHERE task_id = ? AND state = ?",
        )
        .bind(task_id)
        .bind(OPEN)
        .fetch_all(&self.pool)
        .await?;

        for row in rows {
            let id: String = row.try_get("id")?;
            let root = PathBuf::from(row.try_get::<String, _>("root_path")?);
            let repo = PathBuf::from(row.try_get::<String, _>("store_path")?);
            let unsafe_reason: Option<String> = row.try_get("unsafe_reason")?;
            let turn_id: Option<String> = row.try_get("turn_id")?;

            let result = self
                .snapshot_with_timeout(root.clone(), repo, id.clone(), "post")
                .await;
            match result {
                Ok(snapshot) => {
                    let state = if unsafe_reason.is_some() {
                        UNSAFE
                    } else {
                        READY
                    };
                    sqlx::query(
                        "UPDATE filesystem_checkpoints
                         SET post_tree = ?, state = ?, finalized_at = ?
                         WHERE id = ? AND state = ?",
                    )
                    .bind(&snapshot.tree)
                    .bind(state)
                    .bind(Utc::now().to_rfc3339())
                    .bind(&id)
                    .bind(OPEN)
                    .execute(&self.pool)
                    .await?;
                    self.emit(
                        session_id,
                        EventType::CheckpointFinalized,
                        json!({
                            "checkpoint_id": id,
                            "task_id": task_id,
                            "turn_id": turn_id,
                            "root": root,
                            "tree": snapshot.tree,
                            "state": state,
                            "unsafe_reason": unsafe_reason,
                        }),
                    )
                    .await;
                }
                Err(error) => {
                    sqlx::query(
                        "UPDATE filesystem_checkpoints
                         SET state = ?, unsafe_reason = ?, finalized_at = ?
                         WHERE id = ? AND state = ?",
                    )
                    .bind(UNSAFE)
                    .bind(format!("post-task snapshot failed: {error}"))
                    .bind(Utc::now().to_rfc3339())
                    .bind(&id)
                    .bind(OPEN)
                    .execute(&self.pool)
                    .await?;
                    self.emit(
                        session_id,
                        EventType::CheckpointSkipped,
                        json!({
                            "checkpoint_id": id,
                            "task_id": task_id,
                            "turn_id": turn_id,
                            "root": root,
                            "reason": format!("post-task snapshot failed: {error}"),
                        }),
                    )
                    .await;
                }
            }
            self.leases.lock().await.remove(&id);
        }
        self.prune_retention().await
    }

    pub(crate) async fn list_text(&self, limit: usize, root_filter: Option<&str>) -> String {
        match self.list_records(limit.clamp(1, 100)).await {
            Ok(records) => {
                let records = records
                    .into_iter()
                    .filter(|record| {
                        root_filter.is_none_or(|filter| {
                            record.root.to_string_lossy().contains(filter.trim())
                        })
                    })
                    .collect::<Vec<_>>();
                if records.is_empty() {
                    return "No filesystem checkpoints found.".to_string();
                }
                let mut lines = vec![
                    "Filesystem checkpoints (content is stored outside the workspace):".to_string(),
                ];
                for record in records {
                    let unsafe_suffix = record
                        .unsafe_reason
                        .as_deref()
                        .map(|reason| format!(" — {reason}"))
                        .unwrap_or_default();
                    let rollback_suffix = record
                        .rollback_of
                        .as_deref()
                        .map(|id| format!(" safety-for={id}"))
                        .unwrap_or_default();
                    lines.push(format!(
                        "• {} [{}] {} paths / {} bytes — {} — {}{}{}",
                        record.id,
                        record.state,
                        record.included_paths,
                        record.included_bytes,
                        record.created_at,
                        record.root.display(),
                        rollback_suffix,
                        unsafe_suffix
                    ));
                }
                lines.push(
                    "Use /rollback <checkpoint-id> for a conflict preview; confirmation is required."
                        .to_string(),
                );
                lines.join("\n")
            }
            Err(error) => format!("Failed to list filesystem checkpoints: {error}"),
        }
    }

    pub(crate) async fn prepare_rollback(
        &self,
        session_id: &str,
        selector: Option<&str>,
    ) -> anyhow::Result<RollbackPreview> {
        let record = self.select_checkpoint(session_id, selector).await?;
        anyhow::ensure!(
            record.state == READY,
            "checkpoint {} is {} and cannot be rolled back",
            record.id,
            record.state
        );
        let post_tree = record.post_tree.clone().ok_or_else(|| {
            anyhow::anyhow!("checkpoint {} has no finalized post tree", record.id)
        })?;
        let root = record.root.clone();
        let repo = record.repo.clone();
        let pre_tree = record.pre_tree.clone();
        let checkpoint_id = record.id.clone();
        let config = self.config.clone();
        let git = self.git_binary.clone();
        let git_config = self.git_config.clone();
        let plan = tokio::task::spawn_blocking(move || {
            build_rollback_plan(
                &git,
                &git_config,
                &config,
                checkpoint_id,
                root,
                repo,
                pre_tree,
                post_tree,
            )
        })
        .await
        .map_err(|error| anyhow::anyhow!("rollback preview worker failed: {error}"))??;

        let writes = plan
            .operations
            .iter()
            .filter(|operation| operation.restore.is_some())
            .count();
        let deletes = plan.operations.len().saturating_sub(writes);
        let mut samples = plan
            .operations
            .iter()
            .take(8)
            .map(|operation| {
                let action = if operation.restore.is_some() {
                    "restore"
                } else {
                    "delete"
                };
                format!("• {action}: {}", operation.path)
            })
            .collect::<Vec<_>>();
        samples.extend(
            plan.conflicts
                .iter()
                .take(8usize.saturating_sub(samples.len()))
                .map(|conflict| format!("• preserve: {} ({})", conflict.path, conflict.reason)),
        );

        let token = confirmation_token();
        let expires_at = Utc::now() + chrono::Duration::minutes(ROLLBACK_CONFIRMATION_MINUTES);
        let mut confirmations = self.confirmations.lock().await;
        confirmations.retain(|_, confirmation| confirmation.expires_at > Utc::now());
        confirmations.insert(
            token.clone(),
            Confirmation {
                session_id: session_id.to_string(),
                expires_at,
                plan: plan.clone(),
            },
        );
        Ok(RollbackPreview {
            checkpoint_id: plan.checkpoint_id,
            root: plan.root,
            writes,
            deletes,
            conflicts: plan.conflicts.len(),
            samples,
            token,
            expires_at,
        })
    }

    pub(crate) async fn apply_rollback(
        &self,
        session_id: &str,
        token: &str,
    ) -> anyhow::Result<RollbackResult> {
        let confirmation = self
            .confirmations
            .lock()
            .await
            .remove(token.trim())
            .ok_or_else(|| anyhow::anyhow!("unknown or already-used rollback confirmation"))?;
        anyhow::ensure!(
            confirmation.session_id == session_id,
            "rollback confirmation belongs to a different session"
        );
        anyhow::ensure!(
            confirmation.expires_at > Utc::now(),
            "rollback confirmation expired; request a new preview"
        );

        let plan = confirmation.plan;
        let root_lock = self.root_lock(&plan.root).await;
        let _guard = root_lock.lock_owned().await;
        let run_id = checkpoint_id("restore");
        let safety_id = checkpoint_id("cp");
        let safety = self
            .snapshot_with_timeout(
                plan.root.clone(),
                plan.repo.clone(),
                safety_id.clone(),
                "pre",
            )
            .await?;
        self.insert_safety_checkpoint(session_id, &run_id, &safety_id, &plan, &safety)
            .await?;

        let expires = Utc::now() + chrono::Duration::days(self.config.retention_days as i64);
        let plan_json = serde_json::to_string(&plan)?;
        sqlx::query(
            r#"
            INSERT INTO checkpoint_restore_runs
                (id, checkpoint_id, session_id, state, plan_json, next_index,
                 safety_checkpoint_id, created_at, expires_at)
            VALUES (?, ?, ?, 'applying', ?, 0, ?, ?, ?)
            "#,
        )
        .bind(&run_id)
        .bind(&plan.checkpoint_id)
        .bind(session_id)
        .bind(&plan_json)
        .bind(&safety_id)
        .bind(Utc::now().to_rfc3339())
        .bind(expires.to_rfc3339())
        .execute(&self.pool)
        .await?;
        self.emit(
            session_id,
            EventType::RollbackStarted,
            json!({
                "checkpoint_id": plan.checkpoint_id,
                "restore_run_id": run_id,
                "safety_checkpoint_id": safety_id,
                "root": plan.root,
                "operation_count": plan.operations.len(),
                "preview_conflicts": plan.conflicts.len(),
            }),
        )
        .await;

        let mut conflicts = plan.conflicts.clone();
        let mut applied = 0usize;
        for (index, operation) in plan.operations.iter().enumerate() {
            let root = plan.root.clone();
            let repo = plan.repo.clone();
            let operation = operation.clone();
            let operation_path = operation.path.clone();
            let git = self.git_binary.clone();
            let git_config = self.git_config.clone();
            let apply_result = tokio::task::spawn_blocking(move || {
                apply_restore_operation(&git, &git_config, &root, &repo, &operation)
            })
            .await
            .map_err(|error| anyhow::anyhow!("rollback worker failed: {error}"))?;
            match apply_result {
                Ok(true) => applied += 1,
                Ok(false) => conflicts.push(RestoreConflict {
                    path: operation_path.clone(),
                    reason: "path changed after preview".to_string(),
                }),
                Err(error) => {
                    let safety_error = self
                        .finalize_safety_checkpoint(session_id, &safety_id, &plan.root, &plan.repo)
                        .await
                        .err();
                    let mut message = format!("rollback stopped at {operation_path}: {error}");
                    if let Some(safety_error) = safety_error {
                        message.push_str(&format!(
                            "; safety checkpoint finalization also failed: {safety_error}"
                        ));
                    }
                    sqlx::query(
                        "UPDATE checkpoint_restore_runs
                         SET state = 'failed', completed_at = ?, error = ? WHERE id = ?",
                    )
                    .bind(Utc::now().to_rfc3339())
                    .bind(&message)
                    .bind(&run_id)
                    .execute(&self.pool)
                    .await?;
                    self.emit(
                        session_id,
                        EventType::RollbackFailed,
                        json!({
                            "checkpoint_id": plan.checkpoint_id,
                            "restore_run_id": run_id,
                            "safety_checkpoint_id": safety_id,
                            "applied": applied,
                            "error": message,
                        }),
                    )
                    .await;
                    return Err(anyhow::anyhow!(
                        "{message}. Partial changes are recoverable from safety checkpoint {safety_id}"
                    ));
                }
            }
            sqlx::query("UPDATE checkpoint_restore_runs SET next_index = ? WHERE id = ?")
                .bind((index + 1) as i64)
                .bind(&run_id)
                .execute(&self.pool)
                .await?;
        }

        if let Err(error) = self
            .finalize_safety_checkpoint(session_id, &safety_id, &plan.root, &plan.repo)
            .await
        {
            let message =
                format!("rollback applied, but safety checkpoint finalization failed: {error}");
            sqlx::query(
                "UPDATE checkpoint_restore_runs
                 SET state = 'failed', completed_at = ?, error = ? WHERE id = ?",
            )
            .bind(Utc::now().to_rfc3339())
            .bind(&message)
            .bind(&run_id)
            .execute(&self.pool)
            .await?;
            self.emit(
                session_id,
                EventType::RollbackFailed,
                json!({
                    "checkpoint_id": plan.checkpoint_id,
                    "restore_run_id": run_id,
                    "safety_checkpoint_id": safety_id,
                    "applied": applied,
                    "error": message,
                }),
            )
            .await;
            return Err(anyhow::anyhow!(message));
        }

        sqlx::query(
            "UPDATE checkpoint_restore_runs
             SET state = ?, completed_at = ? WHERE id = ?",
        )
        .bind(if conflicts.is_empty() {
            "completed"
        } else {
            "completed_with_conflicts"
        })
        .bind(Utc::now().to_rfc3339())
        .bind(&run_id)
        .execute(&self.pool)
        .await?;
        self.emit(
            session_id,
            EventType::RollbackCompleted,
            json!({
                "checkpoint_id": plan.checkpoint_id,
                "restore_run_id": run_id,
                "safety_checkpoint_id": safety_id,
                "applied": applied,
                "conflicts": conflicts,
            }),
        )
        .await;
        Ok(RollbackResult {
            checkpoint_id: plan.checkpoint_id,
            safety_checkpoint_id: safety_id,
            applied,
            conflicts,
        })
    }

    async fn resolve_root(&self, tool: &str, args: &Value) -> anyhow::Result<PathBuf> {
        anyhow::ensure!(
            self.backend.kind() == BackendKind::Local,
            "filesystem checkpoints currently support only the local execution backend"
        );

        let explicit_path = if matches!(tool, "write_file" | "edit_file") {
            ["path", "file_path", "file", "filename"]
                .iter()
                .find_map(|key| args.get(*key).and_then(Value::as_str))
        } else {
            args.get("working_dir")
                .and_then(Value::as_str)
                .or_else(|| args.get("_project_scope").and_then(Value::as_str))
        };

        let candidate = if let Some(path) = explicit_path {
            PathBuf::from(self.backend.resolve_path(path).await?.as_str())
        } else {
            PathBuf::from(self.backend.workspace_root().as_str())
        };
        let existing = nearest_existing_ancestor(&candidate)
            .ok_or_else(|| anyhow::anyhow!("no existing ancestor for {}", candidate.display()))?;
        let root = fs_utils::find_nearest_project_root(&existing).ok_or_else(|| {
            anyhow::anyhow!(
                "refusing an unbounded checkpoint: {} is not inside a recognized project root",
                candidate.display()
            )
        })?;
        let root = root.canonicalize()?;
        validate_checkpoint_root(&root, &self.store_root)?;
        Ok(root)
    }

    async fn existing_checkpoint_id(
        &self,
        scope_id: &str,
        backend_id: &str,
        root: &Path,
    ) -> anyhow::Result<Option<String>> {
        let row = sqlx::query(
            "SELECT id FROM filesystem_checkpoints
             WHERE scope_id = ? AND backend_id = ? AND root_path = ?",
        )
        .bind(scope_id)
        .bind(backend_id)
        .bind(root.to_string_lossy().as_ref())
        .fetch_optional(&self.pool)
        .await?;
        row.map(|row| row.try_get("id"))
            .transpose()
            .map_err(Into::into)
    }

    async fn root_lock(&self, root: &Path) -> Arc<Mutex<()>> {
        let mut locks = self.root_locks.lock().await;
        locks
            .entry(root.to_path_buf())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    fn ensure_root_repo(&self, root: &Path) -> anyhow::Result<PathBuf> {
        let root_store = self.store_root.join("roots").join(root_key(root));
        fs::create_dir_all(&root_store)?;
        set_owner_only(&root_store)?;
        let repo = root_store.join("repo.git");
        if !repo.join("HEAD").exists() {
            let output = isolated_git_command(&self.git_binary, &self.git_config, None)
                .arg("init")
                .arg("--bare")
                .arg("--quiet")
                .arg(&repo)
                .output()?;
            ensure_success("git init --bare", &output)?;
            set_owner_only(&repo)?;
        }
        Ok(repo)
    }

    async fn snapshot_with_timeout(
        &self,
        root: PathBuf,
        repo: PathBuf,
        checkpoint_id: String,
        phase: &'static str,
    ) -> anyhow::Result<SnapshotStats> {
        let config = self.config.clone();
        let git = self.git_binary.clone();
        let git_config = self.git_config.clone();
        let timeout = Duration::from_secs(config.snapshot_timeout_secs.max(1));
        let repo_for_snapshot = repo.clone();
        let checkpoint_for_snapshot = checkpoint_id.clone();
        let snapshot = tokio::time::timeout(
            timeout,
            tokio::task::spawn_blocking(move || {
                snapshot_tree(
                    &git,
                    &git_config,
                    &config,
                    &root,
                    &repo_for_snapshot,
                    Some((&checkpoint_for_snapshot, phase)),
                )
            }),
        )
        .await
        .map_err(|_| anyhow::anyhow!("checkpoint snapshot exceeded {}s", timeout.as_secs()))?
        .map_err(|error| anyhow::anyhow!("checkpoint snapshot worker failed: {error}"))??;

        let store_root = self.store_root.clone();
        let store_size = tokio::task::spawn_blocking(move || directory_size(&store_root))
            .await
            .map_err(|error| anyhow::anyhow!("checkpoint size worker failed: {error}"))??;
        if store_size > self.config.max_store_bytes {
            let git = self.git_binary.clone();
            let git_config = self.git_config.clone();
            let repo_for_cleanup = repo.clone();
            let checkpoint_for_cleanup = checkpoint_id.clone();
            tokio::task::spawn_blocking(move || {
                delete_checkpoint_ref(
                    &git,
                    &git_config,
                    &repo_for_cleanup,
                    &checkpoint_for_cleanup,
                    phase,
                );
                let _ = isolated_git_command(&git, &git_config, Some(&repo_for_cleanup))
                    .args(["gc", "--prune=now", "--quiet"])
                    .output();
            })
            .await
            .ok();
            anyhow::bail!(
                "checkpoint snapshot would exceed the {} byte store limit",
                self.config.max_store_bytes
            );
        }
        Ok(snapshot)
    }

    async fn insert_safety_checkpoint(
        &self,
        session_id: &str,
        run_id: &str,
        id: &str,
        plan: &RollbackPlan,
        snapshot: &SnapshotStats,
    ) -> anyhow::Result<()> {
        let now = Utc::now();
        let expires = now + chrono::Duration::days(self.config.retention_days as i64);
        sqlx::query(
            r#"
            INSERT INTO filesystem_checkpoints
                (id, scope_id, session_id, backend_id, root_path, store_path,
                 pre_tree, state, origin_tool, included_paths,
                 included_bytes, excluded_paths, rollback_of, created_at,
                 expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'rollback_safety', ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(id)
        .bind(format!("rollback:{run_id}"))
        .bind(session_id)
        .bind(self.backend.id())
        .bind(plan.root.to_string_lossy().as_ref())
        .bind(plan.repo.to_string_lossy().as_ref())
        .bind(&snapshot.tree)
        .bind(OPEN)
        .bind(snapshot.included_paths as i64)
        .bind(snapshot.included_bytes as i64)
        .bind(snapshot.excluded_paths as i64)
        .bind(&plan.checkpoint_id)
        .bind(now.to_rfc3339())
        .bind(expires.to_rfc3339())
        .execute(&self.pool)
        .await?;
        self.emit(
            session_id,
            EventType::CheckpointCreated,
            json!({
                "checkpoint_id": id,
                "name": "rollback_safety",
                "root": plan.root,
                "tree": snapshot.tree,
                "rollback_of": plan.checkpoint_id,
            }),
        )
        .await;
        Ok(())
    }

    async fn finalize_safety_checkpoint(
        &self,
        session_id: &str,
        id: &str,
        root: &Path,
        repo: &Path,
    ) -> anyhow::Result<()> {
        let snapshot = self
            .snapshot_with_timeout(
                root.to_path_buf(),
                repo.to_path_buf(),
                id.to_string(),
                "post",
            )
            .await?;
        sqlx::query(
            "UPDATE filesystem_checkpoints
             SET post_tree = ?, state = ?, finalized_at = ?
             WHERE id = ? AND state = ?",
        )
        .bind(&snapshot.tree)
        .bind(READY)
        .bind(Utc::now().to_rfc3339())
        .bind(id)
        .bind(OPEN)
        .execute(&self.pool)
        .await?;
        self.emit(
            session_id,
            EventType::CheckpointFinalized,
            json!({
                "checkpoint_id": id,
                "name": "rollback_safety",
                "root": root,
                "tree": snapshot.tree,
                "state": READY,
            }),
        )
        .await;
        Ok(())
    }

    async fn list_records(&self, limit: usize) -> anyhow::Result<Vec<CheckpointRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, task_id, turn_id, root_path, store_path,
                   pre_tree, post_tree, state, origin_tool, included_paths,
                   included_bytes, excluded_paths, unsafe_reason, rollback_of,
                   created_at
            FROM filesystem_checkpoints
            ORDER BY created_at DESC LIMIT ?
            "#,
        )
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(record_from_row).collect()
    }

    async fn select_checkpoint(
        &self,
        session_id: &str,
        selector: Option<&str>,
    ) -> anyhow::Result<CheckpointRecord> {
        let selector = selector.unwrap_or("latest").trim();
        let row = if selector.is_empty() || selector.eq_ignore_ascii_case("latest") {
            sqlx::query(
                r#"
                SELECT id, session_id, task_id, turn_id, root_path, store_path,
                       pre_tree, post_tree, state, origin_tool, included_paths,
                       included_bytes, excluded_paths, unsafe_reason, rollback_of,
                       created_at
                FROM filesystem_checkpoints
                WHERE session_id = ? AND state = ?
                ORDER BY created_at DESC LIMIT 1
                "#,
            )
            .bind(session_id)
            .bind(READY)
            .fetch_optional(&self.pool)
            .await?
        } else {
            sqlx::query(
                r#"
                SELECT id, session_id, task_id, turn_id, root_path, store_path,
                       pre_tree, post_tree, state, origin_tool, included_paths,
                       included_bytes, excluded_paths, unsafe_reason, rollback_of,
                       created_at
                FROM filesystem_checkpoints WHERE id = ? LIMIT 1
                "#,
            )
            .bind(selector)
            .fetch_optional(&self.pool)
            .await?
        };
        row.map(record_from_row)
            .transpose()?
            .ok_or_else(|| anyhow::anyhow!("no matching rollback-ready checkpoint found"))
    }

    async fn emit(&self, session_id: &str, event_type: EventType, data: Value) {
        if let Err(error) = self
            .event_store
            .append(Event::new(session_id, event_type, data))
            .await
        {
            warn!(error = %error, "Failed to append checkpoint audit event");
        }
    }

    async fn reconcile_interrupted_state(&self) -> anyhow::Result<()> {
        let now = Utc::now().to_rfc3339();
        let open = sqlx::query(
            "SELECT id, session_id, task_id, turn_id, root_path, store_path, origin_tool
             FROM filesystem_checkpoints WHERE state = ?",
        )
        .bind(OPEN)
        .fetch_all(&self.pool)
        .await?;
        for row in open {
            let id: String = row.try_get("id")?;
            let session_id: String = row.try_get("session_id")?;
            let task_id: Option<String> = row.try_get("task_id")?;
            let turn_id: Option<String> = row.try_get("turn_id")?;
            let root = PathBuf::from(row.try_get::<String, _>("root_path")?);
            let repo = PathBuf::from(row.try_get::<String, _>("store_path")?);
            let origin_tool: String = row.try_get("origin_tool")?;
            if origin_tool == "rollback_safety" {
                match self
                    .finalize_safety_checkpoint(&session_id, &id, &root, &repo)
                    .await
                {
                    Ok(()) => continue,
                    Err(error) => {
                        warn!(
                            checkpoint_id = id,
                            error = %error,
                            "Failed to recover interrupted rollback safety checkpoint"
                        );
                    }
                }
            }
            sqlx::query(
                "UPDATE filesystem_checkpoints
                 SET state = ?, unsafe_reason = ?, finalized_at = ? WHERE id = ?",
            )
            .bind(UNSAFE)
            .bind("daemon restarted before the post-task tree was captured")
            .bind(&now)
            .bind(&id)
            .execute(&self.pool)
            .await?;
            self.emit(
                &session_id,
                EventType::CheckpointFinalized,
                json!({
                    "checkpoint_id": id,
                    "task_id": task_id,
                    "turn_id": turn_id,
                    "root": root,
                    "state": UNSAFE,
                    "unsafe_reason": "daemon restarted before finalization",
                }),
            )
            .await;
        }
        sqlx::query(
            "UPDATE checkpoint_restore_runs
             SET state = 'interrupted', completed_at = ?,
                 error = COALESCE(error, 'daemon restarted during rollback')
             WHERE state = 'applying'",
        )
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn prune_retention(&self) -> anyhow::Result<()> {
        let now = Utc::now();
        let pinned = {
            let mut confirmations = self.confirmations.lock().await;
            confirmations.retain(|_, confirmation| confirmation.expires_at > now);
            confirmations
                .values()
                .map(|confirmation| confirmation.plan.checkpoint_id.clone())
                .collect::<HashSet<_>>()
        };
        let rows = sqlx::query(
            "SELECT id, root_path, store_path, created_at, expires_at
             FROM filesystem_checkpoints
             WHERE state != ? ORDER BY root_path, created_at DESC",
        )
        .bind(OPEN)
        .fetch_all(&self.pool)
        .await?;
        let mut per_root: HashMap<String, u32> = HashMap::new();
        let mut remove = Vec::new();
        for row in rows {
            let id: String = row.try_get("id")?;
            let root: String = row.try_get("root_path")?;
            let count = per_root.entry(root).or_default();
            *count += 1;
            let expires_at: String = row.try_get("expires_at")?;
            let expired = DateTime::parse_from_rfc3339(&expires_at)
                .map(|value| value.with_timezone(&Utc) <= now)
                .unwrap_or(true);
            if !pinned.contains(&id) && (expired || *count > self.config.max_per_root.max(1)) {
                remove.push((id, PathBuf::from(row.try_get::<String, _>("store_path")?)));
            }
        }
        let mut touched_repos = HashSet::new();
        for (id, repo) in remove {
            delete_checkpoint_refs(&self.git_binary, &self.git_config, &repo, &id);
            sqlx::query("DELETE FROM filesystem_checkpoints WHERE id = ?")
                .bind(&id)
                .execute(&self.pool)
                .await?;
            touched_repos.insert(repo);
        }
        for repo in touched_repos {
            let _ = isolated_git_command(&self.git_binary, &self.git_config, Some(&repo))
                .args(["gc", "--prune=now", "--quiet"])
                .output();
        }
        sqlx::query("DELETE FROM checkpoint_restore_runs WHERE expires_at <= ?")
            .bind(now.to_rfc3339())
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}

fn record_from_row(row: sqlx::sqlite::SqliteRow) -> anyhow::Result<CheckpointRecord> {
    Ok(CheckpointRecord {
        id: row.try_get("id")?,
        root: PathBuf::from(row.try_get::<String, _>("root_path")?),
        repo: PathBuf::from(row.try_get::<String, _>("store_path")?),
        pre_tree: row.try_get("pre_tree")?,
        post_tree: row.try_get("post_tree")?,
        state: row.try_get("state")?,
        included_paths: row.try_get::<i64, _>("included_paths")? as usize,
        included_bytes: row.try_get::<i64, _>("included_bytes")? as u64,
        unsafe_reason: row.try_get("unsafe_reason")?,
        rollback_of: row.try_get("rollback_of")?,
        created_at: row.try_get("created_at")?,
    })
}

fn checkpoint_store_path(config: &CheckpointConfig) -> anyhow::Result<PathBuf> {
    if let Some(path) = config.storage_dir.as_deref() {
        let expanded = shellexpand::tilde(path);
        return Ok(PathBuf::from(expanded.as_ref()));
    }
    dirs::data_local_dir()
        .or_else(|| dirs::home_dir().map(|home| home.join(".local").join("share")))
        .map(|base| base.join("aidaemon").join("checkpoints"))
        .ok_or_else(|| anyhow::anyhow!("could not determine a local checkpoint data directory"))
}

fn validate_config(config: &CheckpointConfig) -> anyhow::Result<()> {
    anyhow::ensure!(
        (1..=3650).contains(&config.retention_days),
        "checkpoints.retention_days must be between 1 and 3650"
    );
    anyhow::ensure!(
        (1..=10_000).contains(&config.max_per_root),
        "checkpoints.max_per_root must be between 1 and 10000"
    );
    anyhow::ensure!(
        config.max_store_bytes > 0 && config.max_root_bytes > 0 && config.max_file_bytes > 0,
        "checkpoint byte limits must be greater than zero"
    );
    anyhow::ensure!(
        config.max_file_bytes <= config.max_root_bytes
            && config.max_root_bytes <= config.max_store_bytes,
        "checkpoint size limits must satisfy max_file_bytes <= max_root_bytes <= max_store_bytes"
    );
    anyhow::ensure!(
        (1..=1_000_000).contains(&config.max_paths),
        "checkpoints.max_paths must be between 1 and 1000000"
    );
    anyhow::ensure!(
        (1..=120).contains(&config.snapshot_timeout_secs),
        "checkpoints.snapshot_timeout_secs must be between 1 and 120"
    );
    Ok(())
}

fn validate_store_path(path: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        path.is_absolute(),
        "checkpoints.storage_dir must be an absolute path (a leading ~ is supported)"
    );
    anyhow::ensure!(
        path != Path::new("/"),
        "refusing to use the filesystem root as checkpoint storage"
    );
    if let Some(home) = dirs::home_dir().and_then(|path| path.canonicalize().ok()) {
        let broad = [
            home.clone(),
            home.join("Desktop"),
            home.join("Documents"),
            home.join("Downloads"),
        ];
        anyhow::ensure!(
            !broad.iter().any(|candidate| candidate == path),
            "refusing to use broad personal directory {} as checkpoint storage",
            path.display()
        );
    }
    Ok(())
}

fn discover_git() -> anyhow::Result<PathBuf> {
    let executable = if cfg!(windows) { "git.exe" } else { "git" };
    let path = std::env::var_os("PATH")
        .and_then(|value| {
            std::env::split_paths(&value)
                .map(|directory| directory.join(executable))
                .find(|candidate| candidate.is_file())
        })
        .ok_or_else(|| anyhow::anyhow!("Git is required for filesystem checkpoints"))?;
    path.canonicalize().or(Ok(path))
}

fn isolated_git_command(git: &Path, empty_config: &Path, repo: Option<&Path>) -> Command {
    let mut command = Command::new(git);
    command.env_clear();
    let mut search_paths = vec![
        PathBuf::from("/usr/bin"),
        PathBuf::from("/bin"),
        PathBuf::from("/usr/local/bin"),
        PathBuf::from("/opt/homebrew/bin"),
    ];
    if let Some(parent) = git.parent() {
        search_paths.insert(0, parent.to_path_buf());
    }
    if let Ok(path) = std::env::join_paths(search_paths) {
        command.env("PATH", path);
    }
    command
        .env("HOME", empty_config.parent().unwrap_or(Path::new("/")))
        .env(
            "XDG_CONFIG_HOME",
            empty_config.parent().unwrap_or(Path::new("/")),
        )
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", empty_config)
        .env("GIT_CONFIG_SYSTEM", empty_config)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("LC_ALL", "C");
    if let Some(repo) = repo {
        command.env("GIT_DIR", repo);
    }
    command
}

fn ensure_success(label: &str, output: &Output) -> anyhow::Result<()> {
    if output.status.success() {
        return Ok(());
    }
    anyhow::bail!(
        "{label} failed ({}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    )
}

fn root_key(root: &Path) -> String {
    let mut digest = Sha256::new();
    digest.update(root.to_string_lossy().as_bytes());
    format!("{:x}", digest.finalize())
}

fn checkpoint_id(prefix: &str) -> String {
    let compact = Uuid::new_v4().simple().to_string();
    format!("{prefix}_{}", &compact[..16])
}

fn confirmation_token() -> String {
    Uuid::new_v4().simple().to_string()[..10].to_ascii_uppercase()
}

fn internal_string(args: &Value, key: &str) -> Option<String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn tool_action_can_mutate(tool: &str, arguments: &str) -> bool {
    if matches!(tool, "write_file" | "edit_file" | "run_command") {
        return true;
    }
    if tool == "terminal" {
        let Ok(args) = serde_json::from_str::<Value>(arguments) else {
            return true;
        };
        let action = args.get("action").and_then(Value::as_str).unwrap_or("run");
        if !action.is_empty() && action != "run" {
            return false;
        }
        return args
            .get("command")
            .and_then(Value::as_str)
            .is_none_or(|command| {
                crate::tools::command_semantics::classify_shell_command(command).mutates_state()
            });
    }
    if tool == "cli_agent" {
        return serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|value| {
                value
                    .get("action")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .is_none_or(|action| action.is_empty() || action == "run");
    }
    false
}

fn nearest_existing_ancestor(path: &Path) -> Option<PathBuf> {
    let mut current = path.to_path_buf();
    loop {
        if current.exists() {
            return Some(if current.is_file() {
                current.parent()?.to_path_buf()
            } else {
                current
            });
        }
        if !current.pop() {
            return None;
        }
    }
}

fn validate_checkpoint_root(root: &Path, store_root: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(root.is_absolute(), "checkpoint root must be absolute");
    anyhow::ensure!(root.is_dir(), "checkpoint root must be a directory");
    anyhow::ensure!(
        root != Path::new("/"),
        "refusing to checkpoint the filesystem root"
    );
    anyhow::ensure!(
        !store_root.starts_with(root),
        "checkpoint store cannot be inside the protected workspace"
    );
    if let Some(home) = dirs::home_dir().and_then(|path| path.canonicalize().ok()) {
        let broad = [
            home.clone(),
            home.join("Desktop"),
            home.join("Documents"),
            home.join("Downloads"),
        ];
        anyhow::ensure!(
            !broad.iter().any(|path| path == root),
            "refusing to checkpoint broad personal directory {}",
            root.display()
        );
    }
    Ok(())
}

fn set_owner_only(path: &Path) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn set_owner_only_file(path: &Path) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

fn snapshot_tree(
    git: &Path,
    empty_config: &Path,
    config: &CheckpointConfig,
    root: &Path,
    repo: &Path,
    reference: Option<(&str, &str)>,
) -> anyhow::Result<SnapshotStats> {
    let deadline = Instant::now() + Duration::from_secs(config.snapshot_timeout_secs.max(1));
    let (files, excluded_paths, included_bytes) = collect_snapshot_files(config, root, deadline)?;
    anyhow::ensure!(
        files.len() <= config.max_paths,
        "workspace contains more than {} eligible paths",
        config.max_paths
    );
    anyhow::ensure!(
        included_bytes <= config.max_root_bytes,
        "workspace contains {} eligible bytes, above the {} byte checkpoint limit",
        included_bytes,
        config.max_root_bytes
    );
    if Instant::now() > deadline {
        anyhow::bail!("checkpoint scan exceeded its time limit");
    }

    let mut hash = isolated_git_command(git, empty_config, Some(repo));
    hash.arg("hash-object")
        .arg("-w")
        .arg("--no-filters")
        .arg("--stdin-paths")
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = hash.spawn()?;
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("could not open git hash-object stdin"))?;
        for file in &files {
            stdin.write_all(file.relative.as_bytes())?;
            stdin.write_all(b"\n")?;
        }
    }
    let output = child.wait_with_output()?;
    ensure_success("git hash-object", &output)?;
    let oids = String::from_utf8(output.stdout)?
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    anyhow::ensure!(
        oids.len() == files.len(),
        "git hashed {} paths but {} were expected",
        oids.len(),
        files.len()
    );

    let index = repo
        .parent()
        .unwrap_or(repo)
        .join(format!("index-{}", Uuid::new_v4().simple()));
    let mut read_tree = isolated_git_command(git, empty_config, Some(repo));
    let output = read_tree
        .env("GIT_INDEX_FILE", &index)
        .args(["read-tree", "--empty"])
        .output()?;
    ensure_success("git read-tree --empty", &output)?;

    let update_result = (|| -> anyhow::Result<()> {
        let mut update = isolated_git_command(git, empty_config, Some(repo));
        update
            .env("GIT_INDEX_FILE", &index)
            .args(["update-index", "-z", "--index-info"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = update.spawn()?;
        {
            let stdin = child
                .stdin
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("could not open git update-index stdin"))?;
            for (file, oid) in files.iter().zip(&oids) {
                write!(stdin, "{:o} {}\t{}\0", file.mode, oid, file.relative)?;
            }
        }
        let output = child.wait_with_output()?;
        ensure_success("git update-index", &output)?;
        Ok(())
    })();
    if let Err(error) = update_result {
        let _ = fs::remove_file(&index);
        return Err(error);
    }

    let output = isolated_git_command(git, empty_config, Some(repo))
        .env("GIT_INDEX_FILE", &index)
        .arg("write-tree")
        .output()?;
    let _ = fs::remove_file(&index);
    ensure_success("git write-tree", &output)?;
    let tree = String::from_utf8(output.stdout)?.trim().to_string();
    anyhow::ensure!(!tree.is_empty(), "git write-tree returned no object ID");

    if let Some((checkpoint_id, phase)) = reference {
        let ref_name = format!("refs/aidaemon/checkpoints/{checkpoint_id}/{phase}");
        let output = isolated_git_command(git, empty_config, Some(repo))
            .args(["update-ref", &ref_name, &tree])
            .output()?;
        ensure_success("git update-ref", &output)?;
    }
    Ok(SnapshotStats {
        tree,
        included_paths: files.len(),
        included_bytes,
        excluded_paths,
    })
}

fn collect_snapshot_files(
    config: &CheckpointConfig,
    root: &Path,
    deadline: Instant,
) -> anyhow::Result<(Vec<SnapshotFile>, usize, u64)> {
    let root_for_filter = root.to_path_buf();
    let mut builder = WalkBuilder::new(root);
    builder
        .hidden(false)
        .follow_links(false)
        .standard_filters(true)
        .git_ignore(true)
        .git_global(false)
        .git_exclude(false)
        .parents(false)
        .filter_entry(move |entry| {
            entry.depth() == 0
                || !hard_excluded(
                    entry
                        .path()
                        .strip_prefix(&root_for_filter)
                        .unwrap_or(entry.path()),
                    entry.file_type().is_some_and(|kind| kind.is_dir()),
                )
        });

    let mut files = Vec::new();
    let mut excluded = 0usize;
    let mut total_bytes = 0u64;
    for entry in builder.build() {
        if Instant::now() > deadline {
            anyhow::bail!("checkpoint scan exceeded its time limit");
        }
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                excluded += 1;
                warn!(error = %error, "Skipped unreadable checkpoint path");
                continue;
            }
        };
        if entry.depth() == 0 {
            continue;
        }
        let metadata = match fs::symlink_metadata(entry.path()) {
            Ok(metadata) => metadata,
            Err(_) => {
                excluded += 1;
                continue;
            }
        };
        if !metadata.file_type().is_file() {
            if !metadata.file_type().is_dir() {
                excluded += 1;
            }
            continue;
        }
        let relative = entry.path().strip_prefix(root)?;
        if hard_excluded(relative, false) {
            excluded += 1;
            continue;
        }
        let Some(mut relative) = relative.to_str().map(str::to_string) else {
            excluded += 1;
            continue;
        };
        if relative.contains(['\n', '\r', '\0']) {
            excluded += 1;
            continue;
        }
        if std::path::MAIN_SEPARATOR != '/' {
            relative = relative.replace(std::path::MAIN_SEPARATOR, "/");
        }
        if metadata.len() > config.max_file_bytes {
            excluded += 1;
            continue;
        }
        total_bytes = total_bytes.saturating_add(metadata.len());
        anyhow::ensure!(
            total_bytes <= config.max_root_bytes,
            "workspace exceeds the {} byte checkpoint limit",
            config.max_root_bytes
        );
        files.push(SnapshotFile {
            relative,
            mode: git_file_mode(&metadata),
        });
        anyhow::ensure!(
            files.len() <= config.max_paths,
            "workspace exceeds the {} path checkpoint limit",
            config.max_paths
        );
    }
    files.sort_by(|left, right| left.relative.cmp(&right.relative));
    Ok((files, excluded, total_bytes))
}

fn hard_excluded(relative: &Path, is_dir: bool) -> bool {
    let components = relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str(),
            _ => None,
        })
        .collect::<Vec<_>>();
    let directory_names = [
        ".git",
        ".hg",
        ".svn",
        ".aidaemon",
        "node_modules",
        "target",
        "dist",
        "build",
        ".next",
        ".nuxt",
        ".cache",
        "coverage",
        "__pycache__",
        ".venv",
        "venv",
        "Pods",
        ".gradle",
    ];
    if components
        .iter()
        .any(|component| directory_names.contains(component))
    {
        return true;
    }
    let Some(name) = components.last().copied() else {
        return false;
    };
    if is_dir {
        return matches!(name, ".aws" | ".ssh" | ".gnupg");
    }
    let lower = name.to_ascii_lowercase();
    if lower.starts_with(".env")
        && ![
            ".env.example",
            ".env.sample",
            ".env.template",
            ".env.schema",
        ]
        .contains(&lower.as_str())
    {
        return true;
    }
    if matches!(
        lower.as_str(),
        ".ds_store"
            | ".npmrc"
            | ".pypirc"
            | ".netrc"
            | "credentials.json"
            | "secrets.json"
            | "id_rsa"
            | "id_ed25519"
    ) {
        return true;
    }
    [
        ".log", ".tmp", ".swp", ".swo", ".bak", ".pem", ".key", ".p12", ".pfx", ".jks",
    ]
    .iter()
    .any(|suffix| lower.ends_with(suffix))
        || lower.ends_with('~')
}

fn git_file_mode(metadata: &fs::Metadata) -> u32 {
    #[cfg(unix)]
    {
        if metadata.permissions().mode() & 0o111 != 0 {
            return 0o100755;
        }
    }
    0o100644
}

fn load_tree(
    git: &Path,
    empty_config: &Path,
    repo: &Path,
    tree: &str,
) -> anyhow::Result<BTreeMap<String, TreeEntry>> {
    let output = isolated_git_command(git, empty_config, Some(repo))
        .args(["ls-tree", "-rz", tree])
        .output()?;
    ensure_success("git ls-tree", &output)?;
    let mut entries = BTreeMap::new();
    for record in output.stdout.split(|byte| *byte == 0) {
        if record.is_empty() {
            continue;
        }
        let tab = record
            .iter()
            .position(|byte| *byte == b'\t')
            .ok_or_else(|| anyhow::anyhow!("invalid git ls-tree record"))?;
        let header = std::str::from_utf8(&record[..tab])?;
        let path = std::str::from_utf8(&record[tab + 1..])?.to_string();
        let mut parts = header.split_whitespace();
        let mode = u32::from_str_radix(
            parts
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing tree mode"))?,
            8,
        )?;
        let kind = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing tree object type"))?;
        let oid = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing tree object ID"))?;
        if kind == "blob" {
            entries.insert(
                path,
                TreeEntry {
                    mode,
                    oid: oid.to_string(),
                },
            );
        }
    }
    Ok(entries)
}

#[allow(clippy::too_many_arguments)]
fn build_rollback_plan(
    git: &Path,
    empty_config: &Path,
    config: &CheckpointConfig,
    checkpoint_id: String,
    root: PathBuf,
    repo: PathBuf,
    pre_tree: String,
    post_tree: String,
) -> anyhow::Result<RollbackPlan> {
    let pre = load_tree(git, empty_config, &repo, &pre_tree)?;
    let post = load_tree(git, empty_config, &repo, &post_tree)?;
    let paths = pre
        .keys()
        .chain(post.keys())
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let mut operations = Vec::new();
    let mut conflicts = Vec::new();
    for path in paths {
        let before = pre.get(&path).cloned();
        let after = post.get(&path).cloned();
        if before == after {
            continue;
        }
        let current = match current_entry(&root, &path, config.max_file_bytes) {
            Ok(current) => current,
            Err(error) => {
                conflicts.push(RestoreConflict {
                    path,
                    reason: error.to_string(),
                });
                continue;
            }
        };
        if current == after {
            operations.push(RestoreOp {
                path,
                expected_current: current,
                restore: before,
            });
        } else if current != before {
            conflicts.push(RestoreConflict {
                path,
                reason: "changed independently after the checkpoint".to_string(),
            });
        }
    }
    Ok(RollbackPlan {
        checkpoint_id,
        root,
        repo,
        pre_tree,
        post_tree,
        operations,
        conflicts,
    })
}

fn current_entry(
    root: &Path,
    relative: &str,
    max_file_bytes: u64,
) -> anyhow::Result<Option<TreeEntry>> {
    let path = safe_workspace_path(root, relative)?;
    verify_no_symlink_ancestors(root, &path)?;
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    anyhow::ensure!(
        metadata.file_type().is_file(),
        "current path is not a regular file"
    );
    anyhow::ensure!(
        metadata.len() <= max_file_bytes,
        "current file exceeds the rollback comparison size limit"
    );
    let bytes = fs::read(&path)?;
    let mut digest = Sha1::new();
    digest.update(format!("blob {}\0", bytes.len()).as_bytes());
    digest.update(&bytes);
    Ok(Some(TreeEntry {
        mode: git_file_mode(&metadata),
        oid: format!("{:x}", digest.finalize()),
    }))
}

fn apply_restore_operation(
    git: &Path,
    empty_config: &Path,
    root: &Path,
    repo: &Path,
    operation: &RestoreOp,
) -> anyhow::Result<bool> {
    let current = match current_entry(root, &operation.path, u64::MAX) {
        Ok(current) => current,
        Err(_) => return Ok(false),
    };
    if current != operation.expected_current {
        return Ok(false);
    }
    let path = safe_workspace_path(root, &operation.path)?;
    verify_no_symlink_ancestors(root, &path)?;
    match operation.restore.as_ref() {
        Some(entry) => {
            let output = isolated_git_command(git, empty_config, Some(repo))
                .args(["cat-file", "blob", &entry.oid])
                .output()?;
            ensure_success("git cat-file", &output)?;
            atomic_restore_file(root, &path, &output.stdout, entry.mode)?;
        }
        None => match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        },
    }
    Ok(true)
}

fn safe_workspace_path(root: &Path, relative: &str) -> anyhow::Result<PathBuf> {
    let relative = Path::new(relative);
    anyhow::ensure!(
        !relative.is_absolute()
            && relative
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
        "checkpoint contains an unsafe relative path"
    );
    Ok(root.join(relative))
}

fn verify_no_symlink_ancestors(root: &Path, target: &Path) -> anyhow::Result<()> {
    let relative = target.strip_prefix(root)?;
    let mut current = root.to_path_buf();
    let parent = relative.parent().unwrap_or(Path::new(""));
    for component in parent.components() {
        let Component::Normal(component) = component else {
            anyhow::bail!("unsafe rollback path component");
        };
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                anyhow::bail!("rollback path traverses a symlink")
            }
            Ok(metadata) if !metadata.is_dir() => {
                anyhow::bail!("rollback parent is not a directory")
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn atomic_restore_file(root: &Path, path: &Path, bytes: &[u8], mode: u32) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("rollback path has no parent"))?;
    create_safe_parent_dirs(root, parent)?;
    let temp = parent.join(format!(
        ".aidaemon-rollback-{}.tmp",
        Uuid::new_v4().simple()
    ));
    let result = (|| -> anyhow::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        #[cfg(unix)]
        fs::set_permissions(
            &temp,
            fs::Permissions::from_mode(if mode == 0o100755 { 0o755 } else { 0o644 }),
        )?;
        #[cfg(windows)]
        if path.exists() {
            fs::remove_file(path)?;
        }
        fs::rename(&temp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn create_safe_parent_dirs(root: &Path, parent: &Path) -> anyhow::Result<()> {
    let relative = parent.strip_prefix(root)?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(component) = component else {
            anyhow::bail!("unsafe rollback directory component");
        };
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                anyhow::bail!("rollback parent traverses a symlink")
            }
            Ok(metadata) if metadata.is_dir() => {}
            Ok(_) => anyhow::bail!("rollback parent is not a directory"),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(&current)?;
            }
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn delete_checkpoint_refs(git: &Path, empty_config: &Path, repo: &Path, id: &str) {
    for phase in ["pre", "post"] {
        delete_checkpoint_ref(git, empty_config, repo, id, phase);
    }
}

fn delete_checkpoint_ref(git: &Path, empty_config: &Path, repo: &Path, id: &str, phase: &str) {
    let reference = format!("refs/aidaemon/checkpoints/{id}/{phase}");
    let _ = isolated_git_command(git, empty_config, Some(repo))
        .args(["update-ref", "-d", &reference])
        .output();
}

fn directory_size(path: &Path) -> anyhow::Result<u64> {
    let mut total = 0u64;
    let mut pending = vec![path.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let metadata = fs::symlink_metadata(entry.path())?;
            if metadata.file_type().is_dir() {
                pending.push(entry.path());
            } else if metadata.file_type().is_file() {
                total = total.saturating_add(metadata.len());
            }
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_only_terminal_commands_do_not_request_filesystem_checkpoints() {
        let count_projects = json!({
            "action": "run",
            "command":
                "find \"$HOME/projects\" -mindepth 1 -maxdepth 1 -type d ! -name '.*' -print | wc -l"
        })
        .to_string();
        assert!(!tool_action_can_mutate("terminal", &count_projects));

        let mutating = json!({
            "action": "run",
            "command": "touch generated.txt"
        })
        .to_string();
        assert!(tool_action_can_mutate("terminal", &mutating));
    }

    #[test]
    fn hard_excludes_secrets_builds_and_logs_but_keeps_examples() {
        assert!(hard_excluded(Path::new(".env"), false));
        assert!(hard_excluded(Path::new("nested/.env.local"), false));
        assert!(!hard_excluded(Path::new(".env.example"), false));
        assert!(hard_excluded(Path::new("target/debug/app"), false));
        assert!(hard_excluded(Path::new("node_modules/pkg/index.js"), false));
        assert!(hard_excluded(Path::new("server.log"), false));
        assert!(hard_excluded(Path::new("private.pem"), false));
        assert!(!hard_excluded(Path::new("src/photo.png"), false));
        assert!(!hard_excluded(Path::new("data/app.sqlite"), false));
    }

    #[test]
    fn rollback_rule_preserves_independent_changes() {
        let old = TreeEntry {
            mode: 0o100644,
            oid: "old".into(),
        };
        let agent = TreeEntry {
            mode: 0o100644,
            oid: "agent".into(),
        };
        let user = TreeEntry {
            mode: 0o100644,
            oid: "user".into(),
        };
        assert_eq!(Some(agent.clone()), Some(agent.clone()));
        assert_ne!(Some(user), Some(agent.clone()));
        assert_ne!(Some(old), Some(agent));
    }

    #[test]
    fn safe_workspace_path_rejects_escape() {
        let root = Path::new("/tmp/project");
        assert!(safe_workspace_path(root, "src/main.rs").is_ok());
        assert!(safe_workspace_path(root, "../secret").is_err());
        assert!(safe_workspace_path(root, "/etc/passwd").is_err());
    }

    #[test]
    fn external_shadow_git_round_trip_does_not_create_workspace_git_metadata() {
        let workspace = tempfile::tempdir().unwrap();
        fs::write(workspace.path().join("project.toml"), "marker").unwrap();
        fs::write(workspace.path().join("tracked.txt"), "before").unwrap();
        fs::write(workspace.path().join(".env"), "SECRET=value").unwrap();
        fs::create_dir(workspace.path().join("target")).unwrap();
        fs::write(workspace.path().join("target/output"), "build").unwrap();

        let store = tempfile::tempdir().unwrap();
        let repo = store.path().join("repo.git");
        let empty_config = store.path().join("gitconfig");
        File::create(&empty_config).unwrap();
        let git = discover_git().unwrap();
        let output = isolated_git_command(&git, &empty_config, None)
            .args(["init", "--bare", "--quiet"])
            .arg(&repo)
            .output()
            .unwrap();
        ensure_success("git init", &output).unwrap();
        let config = CheckpointConfig {
            enabled: true,
            storage_dir: None,
            ..CheckpointConfig::default()
        };

        let pre = snapshot_tree(
            &git,
            &empty_config,
            &config,
            workspace.path(),
            &repo,
            Some(("cp_test", "pre")),
        )
        .unwrap();
        fs::write(workspace.path().join("tracked.txt"), "after").unwrap();
        fs::write(workspace.path().join("created.txt"), "new").unwrap();
        let post = snapshot_tree(
            &git,
            &empty_config,
            &config,
            workspace.path(),
            &repo,
            Some(("cp_test", "post")),
        )
        .unwrap();
        let plan = build_rollback_plan(
            &git,
            &empty_config,
            &config,
            "cp_test".into(),
            workspace.path().to_path_buf(),
            repo.clone(),
            pre.tree,
            post.tree,
        )
        .unwrap();
        assert_eq!(plan.operations.len(), 2);
        for operation in &plan.operations {
            assert!(apply_restore_operation(
                &git,
                &empty_config,
                workspace.path(),
                &repo,
                operation
            )
            .unwrap());
        }
        assert_eq!(
            fs::read_to_string(workspace.path().join("tracked.txt")).unwrap(),
            "before"
        );
        assert!(!workspace.path().join("created.txt").exists());
        assert!(!workspace.path().join(".git").exists());
        let tree = load_tree(
            &git,
            &empty_config,
            &repo,
            "refs/aidaemon/checkpoints/cp_test/pre",
        )
        .unwrap();
        assert!(!tree.contains_key(".env"));
        assert!(!tree.contains_key("target/output"));
    }

    #[tokio::test]
    async fn manager_deduplicates_turn_audits_and_requires_confirmed_rollback() {
        let workspace = tempfile::tempdir().unwrap();
        fs::write(
            workspace.path().join("Cargo.toml"),
            "[package]\nname='checkpoint-fixture'\nversion='0.1.0'\n",
        )
        .unwrap();
        let source = workspace.path().join("src.txt");
        fs::write(&source, "before").unwrap();

        let store = tempfile::tempdir().unwrap();
        let pool = SqlitePool::connect("sqlite::memory:").await.unwrap();
        let event_store = Arc::new(EventStore::new(pool.clone()).await.unwrap());
        let config = CheckpointConfig {
            enabled: true,
            storage_dir: Some(store.path().to_string_lossy().into_owned()),
            ..CheckpointConfig::default()
        };
        let manager = CheckpointManager::new(
            config,
            pool.clone(),
            event_store,
            crate::execution::active_execution_backend(),
        )
        .await
        .unwrap();
        let args = json!({
            "path": source,
            "_session_id": "telegram:owner",
            "_task_id": "task-1",
            "_turn_id": "turn-1"
        })
        .to_string();

        let first = manager
            .begin_for_tool("write_file", &args)
            .await
            .unwrap()
            .unwrap();
        let duplicate = manager
            .begin_for_tool("edit_file", &args)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(first, duplicate);
        fs::write(&source, "after").unwrap();
        manager
            .finalize_task("task-1", "telegram:owner")
            .await
            .unwrap();

        let row = sqlx::query("SELECT state, post_tree FROM filesystem_checkpoints WHERE id = ?")
            .bind(&first)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(row.get::<String, _>("state"), READY);
        assert!(!row.get::<String, _>("post_tree").is_empty());
        let audit_count = sqlx::query(
            "SELECT COUNT(*) AS count FROM events
             WHERE event_type IN ('checkpoint_created', 'checkpoint_finalized')",
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .get::<i64, _>("count");
        assert_eq!(audit_count, 2);

        let preview = manager
            .prepare_rollback("telegram:owner", Some(&first))
            .await
            .unwrap();
        assert_eq!(preview.writes, 1);
        assert_eq!(fs::read_to_string(&source).unwrap(), "after");
        let result = manager
            .apply_rollback("telegram:owner", &preview.token)
            .await
            .unwrap();
        assert_eq!(result.applied, 1);
        assert_eq!(fs::read_to_string(&source).unwrap(), "before");
        let safety_preview = manager
            .prepare_rollback("telegram:owner", Some(&result.safety_checkpoint_id))
            .await
            .unwrap();
        assert_eq!(safety_preview.writes, 1);
        manager
            .apply_rollback("telegram:owner", &safety_preview.token)
            .await
            .unwrap();
        assert_eq!(fs::read_to_string(&source).unwrap(), "after");
        assert!(!workspace.path().join(".git").exists());
        let rollback_events = sqlx::query(
            "SELECT COUNT(*) AS count FROM events
             WHERE event_type IN ('rollback_started', 'rollback_completed')",
        )
        .fetch_one(&pool)
        .await
        .unwrap()
        .get::<i64, _>("count");
        assert_eq!(rollback_events, 4);

        let second_args = json!({
            "path": source,
            "_session_id": "telegram:owner",
            "_task_id": "task-2",
            "_turn_id": "turn-2"
        })
        .to_string();
        let second = manager
            .begin_for_tool("write_file", &second_args)
            .await
            .unwrap()
            .unwrap();
        fs::write(&source, "agent-second-change").unwrap();
        manager
            .finalize_task("task-2", "telegram:owner")
            .await
            .unwrap();
        let conflict_preview = manager
            .prepare_rollback("telegram:owner", Some(&second))
            .await
            .unwrap();
        fs::write(&source, "independent-user-change").unwrap();
        let conflict_result = manager
            .apply_rollback("telegram:owner", &conflict_preview.token)
            .await
            .unwrap();
        assert_eq!(conflict_result.applied, 0);
        assert_eq!(conflict_result.conflicts.len(), 1);
        assert_eq!(
            fs::read_to_string(&source).unwrap(),
            "independent-user-change"
        );
    }
}
