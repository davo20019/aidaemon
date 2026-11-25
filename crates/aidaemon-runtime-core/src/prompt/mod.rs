//! The system prompt: the cacheable core (`core`), its fixed rule blocks
//! (`rules`) and the prefix fingerprints that attribute cache breaks to a
//! region of the prompt (`fingerprint`), and the turn's system prompt
//! built on its host (`system`).

pub mod core;
pub mod fingerprint;
pub mod rules;
pub mod system;
