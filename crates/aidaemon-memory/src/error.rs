use aidaemon_core::ProviderError;

/// Errors from this crate.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// A [`crate::MemoryStore`] failed. Custom stores report their own
    /// failures through this variant.
    #[error("memory store: {0}")]
    Store(String),
    #[error("sqlite: {0}")]
    Sqlx(#[from] sqlx::Error),
    #[error(
        "memory schema version {found} is newer than this build supports ({supported}); \
         upgrade aidaemon-memory or use a different database file"
    )]
    SchemaTooNew { found: i64, supported: i64 },
    #[error("memory migration {version} ({name}) failed: {source}")]
    Migration {
        version: i64,
        name: &'static str,
        source: sqlx::Error,
    },
    /// An [`crate::Embedder`] failed.
    #[error("embedding: {0}")]
    Embedding(String),
    #[error(transparent)]
    Provider(#[from] ProviderError),
    /// The input was rejected before it reached the store.
    #[error("invalid memory input: {0}")]
    Invalid(String),
    /// A writer was asked to do work it has no component for (for example
    /// `ingest_run` without a `FactExtractor`).
    #[error("not configured: {0}")]
    NotConfigured(&'static str),
}

pub type Result<T, E = MemoryError> = std::result::Result<T, E>;
