//! # aidaemon-memory (experimental)
//!
//! Long-term memory for agents built on `aidaemon-core`, ported from the
//! daemon's memory system: durable **facts** keyed by
//! `(scope, category, key)` with supersession instead of overwrite,
//! **episodes** summarizing past sessions, hybrid retrieval (semantic +
//! lexical + freshness + confidence + provenance), recall decay and
//! retention.
//!
//! Writing is model-driven: [`FactExtractor`] and [`EpisodeSummarizer`] ask
//! any `aidaemon_core::Provider` for a typed JSON payload. Nothing decides
//! what to remember from phrase lists, and retrieval is scoring with
//! relevance thresholds, not keyword gates.
//!
//! ## Integration
//!
//! The rest of the SDK never depends on this crate. It plugs in through
//! public traits only:
//!
//! | Piece | Plugs in as |
//! |---|---|
//! | [`full::FullMemory`] | `aidaemon_policies::MemorySystem`: aidaemon's own memory (facts, episodes, procedures, people, profile) over [`sqlite::SqliteMemory`], with its tools and run-end writer; the SDK preset's default |
//! | [`ScopedMemory`] | `aidaemon_policies::MemorySystem`: a small scope-keyed memory over any [`MemoryStore`] |
//! | [`MemoryRecall`] | `aidaemon_policies::Recall`, for `MemoryInjection` (`before_model`) |
//! | [`MemoryWriter`] | `aidaemon_core::Hook` (`on_run_end`, background extraction) |
//! | [`RememberTool`], [`RecallTool`] | `aidaemon_core::Tool` |
//! | [`Memory::ingest_run`] | explicit call with a `RunOutcome` |
//!
//! ## Example
//!
//! ```no_run
//! # async fn demo() -> Result<(), aidaemon_memory::MemoryError> {
//! use aidaemon_memory::{HashEmbedder, Memory, MemoryRecall, NewFact, SqliteMemoryStore};
//!
//! let store = SqliteMemoryStore::open("memory.db").await?;
//! let memory = Memory::new(store, HashEmbedder::default());
//!
//! let scope = "telegram:synthetic-user-1";
//! memory
//!     .remember(NewFact::new(scope, "person", "daughters", "Mia and Zoe"))
//!     .await?;
//! for hit in memory.recall_facts(scope, "what are my daughters called", 5).await? {
//!     println!("{:.2} {}", hit.score, hit.fact.value);
//! }
//!
//! // Automatic injection into every model request:
//! let injection = aidaemon_policies::MemoryInjection::new(MemoryRecall::new(memory.clone()));
//! # let _ = injection;
//! # Ok(()) }
//! ```
//!
//! ## Bring your own memory
//!
//! [`Memory`] holds only trait objects, so every part is swappable without
//! touching this crate:
//!
//! - [`MemoryStore`]: storage (pgvector, Qdrant, a document DB, ...). Only
//!   row storage is required; lexical scoring has a default, and overriding
//!   [`MemoryStore::search_facts`] / [`MemoryStore::search_episodes`] lets a
//!   backend rank natively and bypass the scorer. Access levels, the graph
//!   signal and vector refresh are optional methods with safe defaults: a
//!   store without them hides its records from shared channels, scores with
//!   a graph signal of 0, and keeps old-model rows lexical-only.
//! - [`Embedder`]: an embedding API or local model ([`HashEmbedder`] for
//!   tests and offline use, `FastEmbedder` behind the `fastembed` feature).
//! - [`Scorer`]: ranking ([`HybridScorer`], the daemon's weights, by
//!   default).
//! - Any `aidaemon_core::Provider` for [`FactExtractor`] and
//!   [`EpisodeSummarizer`].
//!
//! Or replace this crate entirely: implement `aidaemon_policies::MemorySystem`
//! (the hooks and tools memory adds to a turn) or just
//! `aidaemon_policies::Recall` (snippets for injection) in your own crate.
//! The SDK only ever sees those traits.

#[cfg(feature = "fastembed")]
mod background_model;
#[cfg(feature = "conformance")]
pub mod conformance;
pub mod consolidation;
mod embed;
pub mod embedding_index;
/// The daemon's local embedder and reranker (fastembed), or deterministic
/// offline vectors: the caller picks one by constructor.
#[cfg(feature = "fastembed")]
pub mod embeddings;
pub mod episodes;
mod error;
mod extract;
pub mod full;
pub mod history;
mod learning;
mod memory;
pub mod neighborhood;
pub mod people_intelligence;
pub mod policy;
pub mod profile;
pub mod progressive;
mod recall;
pub mod records;
pub mod render;
pub mod retrieval_eval;
mod scoring;
pub mod search;
#[cfg(test)]
mod search_fixture;
pub mod skills;
pub mod sqlite;
mod store;
pub mod stores;
mod system;
pub mod task_learning;
mod text;
mod tools;
pub mod trajectory;
mod types;
mod writer;

#[cfg(feature = "fastembed")]
pub use embed::FastEmbedder;
pub use embed::{
    Embedder, HashEmbedder, cosine, decode_embedding, decode_embedding_with_dim, encode_embedding,
};
pub use error::{MemoryError, Result};
pub use extract::{
    EpisodeDraft, EpisodeSummarizer, ExtractedFact, Extraction, FactExtractor, parse_facts,
};
pub use learning::{
    BehaviorPattern, ErrorSolution, Expertise, LearningStore, Procedure, UserProfile,
    backfill_learning_embeddings, detect_domains, error_pattern, expertise_level,
    extract_trigger_pattern, generalize_procedure, new_error_solution, new_procedure,
    procedure_base_name, procedure_keyed_name, record_tool_sequence_patterns, summarize_solution,
    tool_sequence_patterns, update_communication_profile, verbosity_from_messages,
};
pub use memory::{
    AudienceResolver, IngestReport, MAX_REEMBED_BATCH, Memory, MemoryBuilder, ScopeResolver,
};
pub use recall::MemoryRecall;
pub use records::FactPrivacy;
pub use scoring::{
    HybridScorer, HybridWeights, Scorer, ScoringConfig, episode_importance, episode_recall_boost,
    term_overlap, text_similarity, tokens, turn_importance, word_overlap,
};
pub use store::{
    InMemoryMemoryStore, MIGRATIONS, MemoryStore, Migration, SqliteMemoryStore, latest_version,
    related_fact_scores,
};
pub use system::ScopedMemory;
pub use tools::{RecallTool, RememberTool};
pub use types::{
    DecayReport, Episode, EpisodeRecord, Fact, FactAccess, FactRecord, MAX_FACT_VALUE_CHARS,
    MemoryStats, MergeConfig, NewEpisode, NewFact, Provenance, RecallAudience, RecordKind,
    ReembedCursor, ReembedReport, RetentionConfig, Score, ScoredEpisode, ScoredFact, Signals,
    StaleEmbedding, StoredEmbedding, UpsertOutcome, episode_search_text, fact_search_text,
    normalize_category, normalize_key, same_value,
};
pub use writer::{MemoryWriter, WriterReport};
