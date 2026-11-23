//! `spawn_agent` lives in `aidaemon-runtime` ([`aidaemon_runtime::spawn_tool`]),
//! with its tests.

pub use aidaemon_runtime::spawn_tool::*;

// Stays here: it checks the defaults of the daemon's own `[subagents]` config.
#[cfg(test)]
mod tests {
    #[test]
    fn config_defaults() {
        use crate::config::SubagentsConfig;
        let cfg = SubagentsConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.max_depth, 3);
        assert_eq!(cfg.max_iterations, 10);
        assert_eq!(cfg.max_response_chars, 8000);
        assert_eq!(cfg.timeout_secs, 300);
    }
}
