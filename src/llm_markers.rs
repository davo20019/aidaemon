// Shared marker constants used in prompts, sanitizers, and provider shims.

#[cfg(test)]
pub(crate) const TEXT_ONLY_RESPONSE_MARKER: &str = "[TEXT_ONLY_RESPONSE_MODE]";
#[cfg(test)]
pub(crate) use aidaemon_runtime::markers::INTENT_GATE_MARKER;
