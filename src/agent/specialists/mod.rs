//! Specialist registry: loads per-kind expert profiles (system prompt, tools,
//! model, budgets) from bundled `.md` files plus optional user overrides.
//!
//! Public surface lives here; private parsing/rendering helpers will be split
//! into submodules as the module grows.

#![allow(dead_code)] // populated by subsequent tasks
