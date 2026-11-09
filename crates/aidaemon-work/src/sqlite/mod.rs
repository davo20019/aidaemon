//! The SQLite implementation of the work stores, over a host's pool.
//!
//! The store traits are implemented for anything that is a [`WorkDatabase`],
//! so a host that already owns the database implements that and gets the
//! stores; [`SqliteWorkStore`] is the ready-made one.

use std::sync::Arc;

use async_trait::async_trait;
use sqlx::{Row, SqlitePool};

use crate::traits::{
    Goal, GoalSchedule, GoalTokenBudgetStatus, ScheduledRunState, Task, TaskActivity,
};

mod goals;
mod mandates;
mod notifications;
mod work;

pub use mandates::enforce_mandate_invariants_on_startup;

/// What the work stores need from whoever owns the database.
pub trait WorkDatabase: Send + Sync {
    /// The pool every statement runs on.
    fn work_pool(&self) -> &SqlitePool;

    /// Strip secrets from text before it is kept in activity logs, journals
    /// and handoffs.
    fn redact_secrets(&self, text: &str) -> String;
}

/// A [`WorkDatabase`] on its own pool, for hosts with no store of their own.
pub struct SqliteWorkStore {
    pool: SqlitePool,
    redact: Arc<dyn Fn(&str) -> String + Send + Sync>,
}

impl SqliteWorkStore {
    /// A store over `pool`, which already has the tables of [`crate::schema`].
    pub fn new(pool: SqlitePool, redact: Arc<dyn Fn(&str) -> String + Send + Sync>) -> Self {
        Self { pool, redact }
    }

    pub fn pool(&self) -> SqlitePool {
        self.pool.clone()
    }
}

impl WorkDatabase for SqliteWorkStore {
    fn work_pool(&self) -> &SqlitePool {
        &self.pool
    }

    fn redact_secrets(&self, text: &str) -> String {
        (self.redact)(text)
    }
}

#[cfg(test)]
pub(crate) async fn open_for_tests(path: &str) -> SqliteWorkStore {
    use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
    let options = SqliteConnectOptions::new()
        .filename(path)
        .create_if_missing(true)
        .journal_mode(SqliteJournalMode::Delete)
        .busy_timeout(std::time::Duration::from_secs(30));
    let pool = SqlitePoolOptions::new()
        .max_connections(10)
        .acquire_timeout(std::time::Duration::from_secs(30))
        .connect_with(options)
        .await
        .unwrap();
    aidaemon_events::schema::migrate_events(&pool)
        .await
        .unwrap();
    // goals.source_episode_id refers to long-term memory's table.
    sqlx::query("CREATE TABLE episodes (id INTEGER PRIMARY KEY)")
        .execute(&pool)
        .await
        .ok();
    crate::schema::migrate_work_tables(&pool, |_| false)
        .await
        .unwrap();
    crate::schema::migrate_work_tail(&pool).await.unwrap();
    SqliteWorkStore::new(pool, Arc::new(|text: &str| text.to_string()))
}
