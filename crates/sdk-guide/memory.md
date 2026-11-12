# Swapping memory

Long-term memory is the `aidaemon-memory` crate (feature `memory` on the SDK).
It is optional and the rest of the SDK never depends on it: it plugs in
through public traits. `Memory` holds trait objects only, so each part is
replaceable:

- `MemoryStore`: where facts and episodes live (SQLite is included; implement
  the trait for pgvector, Qdrant, a document DB, ...). Only row storage is
  required; other methods have safe defaults.
- `Embedder`: turns text into vectors (`model_id`, `dimensions`, `embed`).
  `HashEmbedder` works offline; the `fastembed` feature adds local ONNX models.

```rust,ignore
use aidaemon_memory::{HashEmbedder, Memory, MemoryRecall, NewFact, SqliteMemoryStore};

let store = SqliteMemoryStore::open("memory.db").await?;
let memory = Memory::new(store, HashEmbedder::default());

memory
    .remember(NewFact::new("user:1", "person", "daughters", "Mia and Zoe"))
    .await?;
let hits = memory.recall_facts("user:1", "what are my daughters called", 5).await?;

// Inject recalled facts into every model request:
let hook = aidaemon_policies::MemoryInjection::new(MemoryRecall::new(memory.clone()));
// builder.hook(hook)
```

To swap the backend, implement `MemoryStore` and pass it to `Memory::new`
instead of `SqliteMemoryStore`; nothing else changes. Run the conformance kit
against your implementation (feature `conformance`):
`aidaemon_memory::conformance::check_store(|| async { /* a fresh store */ })`
returns a report of which behaviours it satisfies.

Writing is model-driven (`FactExtractor`, `EpisodeSummarizer` ask any
`Provider` for typed JSON); there are also `RememberTool` and `RecallTool` so
the agent can decide for itself what to store.
