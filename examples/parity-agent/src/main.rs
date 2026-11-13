//! Parity: the preset assembled from the SDK crates alone, run through the
//! shared behavior suite (scripted model, in-process channel) with three
//! memory systems: aidaemon's full memory (the default), `ScopedMemory`, and
//! a notebook written here. The full memory's own cases are skipped for
//! the other two; everything else must pass for all three.
//!
//! cargo run -p aidaemon-parity-agent
use std::sync::Arc;

use aidaemon_sdk::memory::{HashEmbedder, Memory, ScopedMemory, SqliteMemoryStore};
use aidaemon_sdk::policies::MemoryTurn;
use aidaemon_sdk::testing::behavior::{self, BoxError, Parts, Subject};
use aidaemon_sdk::{Agent, preset};

mod notebook;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
enum Kind {
    Full,
    Scoped,
    Notebook,
}

/// The assembly under test: the preset, with one memory system.
async fn assemble(parts: Parts, kind: Kind) -> Result<Agent, BoxError> {
    let dir = parts.memory_dir().to_path_buf();
    let turn = MemoryTurn {
        explicit_writes_only: true, // the scripted model has no turn for extraction
        ..MemoryTurn::owner(parts.id())
    };
    let agent = preset(parts.id()).instructions("You are a helpful assistant. Keep answers short.");
    let agent = match kind {
        Kind::Full => agent.memory_path(dir.join("memory.db")),
        Kind::Scoped => {
            let store = Arc::new(SqliteMemoryStore::open(dir.join("scoped.db")).await?);
            let memory = Memory::builder(store, Arc::new(HashEmbedder::default()));
            agent.memory(Arc::new(ScopedMemory::new(memory)), turn)
        }
        Kind::Notebook => agent.memory(Arc::new(notebook::Notebook::open(&dir)), turn),
    };
    Ok(parts.finish(agent).build().await?)
}

fn subjects() -> [Subject; 3] {
    [
        Subject::new("FullMemory (default)", |p| assemble(p, Kind::Full)).full_memory(),
        Subject::new("ScopedMemory", |p| assemble(p, Kind::Scoped)),
        Subject::new("Notebook (custom)", |p| assemble(p, Kind::Notebook)),
    ]
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let mut passed = true;
    for subject in subjects() {
        let report = behavior::run(&subject).await;
        print!("{report}");
        passed &= report.all_passed();
    }
    if !passed {
        return Err("the behavior suite failed".into());
    }
    Ok(())
}
