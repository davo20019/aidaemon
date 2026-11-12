//! The full memory system: aidaemon's own long-term memory over
//! [`SqliteMemory`](crate::sqlite::SqliteMemory). It recalls facts,
//! episodes, procedures, people and a core profile into each turn, scoped by
//! the turn's audience. A host supplies what is its own (goals, settings,
//! diagnostics) through small traits.

pub mod recall;
pub mod system;
pub mod tools;
pub mod writer;

pub use recall::{
    FullRecall, RecallDecision, RecallHost, RecallSettings, RecallSource, RecallTurn,
    IDENTITY_CATEGORIES,
};
pub use system::{FullMemory, FullTurn};
