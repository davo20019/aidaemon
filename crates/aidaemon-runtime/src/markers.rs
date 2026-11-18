//! Marker constants shared by prompts, sanitizers and provider shims.

/// The envelope of the retired intent gate; never valid in a final reply.
pub const INTENT_GATE_MARKER: &str = "[INTENT_GATE]";
