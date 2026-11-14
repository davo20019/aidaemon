# aidaemon-memory

**Tier: public contract.** Semver-checked in CI; breaking changes only in a new minor version while 0.x, with a migration note.

Long-term memory for agents: durable facts with supersession, episode summaries and hybrid recall. The `MemoryStore` and `Embedder` traits are the public contract, so storage and embeddings are swappable; a SQLite store and an offline `HashEmbedder` are included (`fastembed` feature adds local ONNX embeddings).

```rust,ignore
use aidaemon_memory::{HashEmbedder, Memory, NewFact, SqliteMemoryStore};

let memory = Memory::new(SqliteMemoryStore::open("memory.db").await?, HashEmbedder::default());
memory.remember(NewFact::new("user:1", "person", "daughters", "Mia and Zoe")).await?;
let hits = memory.recall_facts("user:1", "what are my daughters called", 5).await?;
```

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.20.0). MIT licensed. Docs: https://docs.rs/aidaemon-memory
