//! Pure validators for specialist definitions.
//!
//! These helpers clamp untrusted/declared specialist parameters into the
//! ranges enforced by the spawn flow in `agent/runtime/spawn.rs`.

use aidaemon_agents::specialist::SpecialistKind;
use tracing::{debug, warn};

/// Intersect a specialist's tool allowlist with the set of tools its role is
/// permitted to use. Tools missing from this run are dropped; out-of-scope
/// tools are dropped with a warn.
///
/// A declared tool can be missing for ordinary reasons: config disables it
/// (`[tools].disabled`), or the surface does
/// not offer it. That is logged at debug, not warn, so it does not repeat on
/// every spawn. Misspelled names in the bundled specialists are caught by
/// `bundled_specialists_name_only_real_tools` instead.
pub fn intersect_tools(
    kind: SpecialistKind,
    declared: &[String],
    role_scope: &[&str],
    known_tools: &[&str],
) -> Vec<String> {
    let mut out = Vec::new();
    for name in declared {
        if !known_tools.contains(&name.as_str()) {
            debug!(
                kind = %kind.as_str(),
                tool = %name,
                "specialist tool not available in this run — dropped"
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

    /// Every tool a bundled specialist names is a tool some source defines, so
    /// a dropped tool at spawn means "not available here", never a typo.
    #[test]
    fn bundled_specialists_name_only_real_tools() {
        // The workspace root: this crate sits at `crates/aidaemon-runtime`.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut sources = String::new();
        let mut stack = vec![root.join("src"), root.join("crates")];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    sources.push_str(&std::fs::read_to_string(&path).unwrap_or_default());
                }
            }
        }
        let mut checked = 0;
        for entry in std::fs::read_dir(root.join("specialists"))
            .unwrap()
            .flatten()
        {
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "md") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            let frontmatter = text.split("---").nth(1).unwrap_or_default();
            let mut in_tools = false;
            for line in frontmatter.lines() {
                if line.starts_with("tools:") {
                    in_tools = true;
                    continue;
                }
                if !line.starts_with("  ") {
                    in_tools = false;
                }
                let Some(tool) = line.trim().strip_prefix("- ").filter(|_| in_tools) else {
                    continue;
                };
                assert!(
                    sources.contains(&format!("\"name\": \"{tool}\"")),
                    "{} names tool `{tool}`, which no tool schema defines",
                    path.display()
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "no specialist tools were checked");
    }

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
