//! Pure validators for specialist definitions.
//!
//! These helpers clamp untrusted/declared specialist parameters into the
//! ranges enforced by the spawn flow. They are consumed by Task 12's
//! spawn-flow integration; until then they are dead from production code's
//! perspective but exercised by unit tests.

use crate::traits::SpecialistKind;
use tracing::warn;

/// Intersect a specialist's tool allowlist with the set of tools its role is
/// permitted to use. Unknown or out-of-scope tools are dropped with a warn.
#[allow(dead_code)] // consumed by spawn flow in Task 12
pub fn intersect_tools(
    kind: SpecialistKind,
    declared: &[String],
    role_scope: &[&str],
    known_tools: &[&str],
) -> Vec<String> {
    let mut out = Vec::new();
    for name in declared {
        if !known_tools.contains(&name.as_str()) {
            warn!(
                kind = %kind.as_str(),
                tool = %name,
                "specialist references unknown tool — dropped"
            );
            continue;
        }
        if !role_scope.contains(&name.as_str()) {
            warn!(
                kind = %kind.as_str(),
                tool = %name,
                "specialist tool outside role scope — dropped"
            );
            continue;
        }
        if !out.contains(name) {
            out.push(name.clone());
        }
    }
    out
}

/// Clamp a specialist's `max_iterations` into `[1, cap]`.
#[allow(dead_code)] // consumed by spawn flow in Task 12
pub fn clamp_max_iterations(kind: SpecialistKind, declared: usize, cap: usize) -> usize {
    if declared == 0 {
        warn!(kind = %kind.as_str(), "specialist max_iterations=0 → clamped to 1");
        return 1;
    }
    if declared > cap {
        warn!(
            kind = %kind.as_str(),
            declared,
            cap,
            "specialist max_iterations exceeds cap → clamped"
        );
        return cap;
    }
    declared
}

/// Clamp a specialist's `timeout_secs` into `[1, cap]`.
#[allow(dead_code)] // consumed by spawn flow in Task 12
pub fn clamp_timeout(kind: SpecialistKind, declared: u64, cap: u64) -> u64 {
    if declared == 0 {
        warn!(kind = %kind.as_str(), "specialist timeout_secs=0 → clamped to 1");
        return 1;
    }
    declared.min(cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersect_drops_unknown_tools() {
        let out = intersect_tools(
            SpecialistKind::Code,
            &["read_file".to_string(), "bogus_tool".to_string()],
            &["read_file", "write_file"],
            &["read_file", "write_file"],
        );
        assert_eq!(out, vec!["read_file".to_string()]);
    }

    #[test]
    fn intersect_drops_tools_outside_role_scope() {
        let out = intersect_tools(
            SpecialistKind::Code,
            &["read_file".to_string(), "manage_goal_tasks".to_string()],
            &["read_file", "write_file"],
            &["read_file", "write_file", "manage_goal_tasks"],
        );
        assert_eq!(out, vec!["read_file".to_string()]);
    }

    #[test]
    fn intersect_deduplicates() {
        let out = intersect_tools(
            SpecialistKind::Code,
            &["read_file".to_string(), "read_file".to_string()],
            &["read_file"],
            &["read_file"],
        );
        assert_eq!(out, vec!["read_file".to_string()]);
    }

    #[test]
    fn clamp_max_iterations_low_and_high() {
        assert_eq!(clamp_max_iterations(SpecialistKind::Code, 0, 100), 1);
        assert_eq!(clamp_max_iterations(SpecialistKind::Code, 50, 100), 50);
        assert_eq!(clamp_max_iterations(SpecialistKind::Code, 1000, 100), 100);
    }

    #[test]
    fn clamp_timeout_low_and_high() {
        assert_eq!(clamp_timeout(SpecialistKind::Code, 0, 600), 1);
        assert_eq!(clamp_timeout(SpecialistKind::Code, 500, 600), 500);
        assert_eq!(clamp_timeout(SpecialistKind::Code, 9999, 600), 600);
    }
}
