//! The system prompt: the cacheable core (`core`), its fixed rule blocks
//! (`rules`) and the prefix fingerprints that attribute cache breaks to a
//! region of the prompt (`fingerprint`).

pub mod core;
pub mod fingerprint;
pub mod rules;
