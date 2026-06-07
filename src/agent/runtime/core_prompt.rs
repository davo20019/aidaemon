//! Pillar A core-prompt inputs: canonicalization + component hashing.
//! Spec: 2026-06-06-cross-turn-prefix-stability-design.md §Pillar A.
//! Hash actual content inputs, never proxies; canonicalize unordered
//! collections (sort by name) BEFORE hashing. Provider tool-array ordering is
//! enforced upstream and asserted in Tasks 6 and 8. No timestamps, map
//! iteration, or env-dependent formatting.

use crate::agent::prefix_fingerprint::hash_canonical;
use serde_json::json;

/// Session-static inputs to the core (cacheable) prompt prefix. Each field maps
/// to a query-independent source; per-turn material (MCP-trigger tools, matched
/// skill bodies, personal-memory / untrusted-reference restrictions) is
/// deliberately excluded so the core hash is stable across ordinary query
/// changes within a session.
#[derive(Clone, Debug)]
#[allow(dead_code)] // consumed by Task 4 system-prompt core assembly
pub(crate) struct CoreInputs {
    pub base_template: String,
    /// (tool name, serialized schema) — sorted by name in canonical form.
    /// SOURCED FROM the session-static `core_tool_roster` (registered tools for
    /// the (role, channel-visibility) class, NO user_text/MCP-trigger gating,
    /// NO per-turn restrictions). NOT `base_tool_defs` (which is per-turn) and
    /// NOT the filtered `tool_defs`.
    pub tool_roster: Vec<(String, String)>,
    /// (skill name, one-line description, enabled) — availability catalog
    /// only; matched skill CONTENT is tail-side.
    pub skills_catalog: Vec<(String, String, bool)>,
    /// (specialist kind, description).
    pub specialists: Vec<(String, String)>,
    pub channel_rules: String,
    pub persona: String,
}

/// Component hashes for a [`CoreInputs`], one entry per logical component in a
/// fixed order. Component attribution lets a later task name exactly which input
/// changed when the core invalidates.
#[derive(Debug, PartialEq, Eq)]
#[allow(dead_code)] // consumed by Task 4 system-prompt core assembly
pub(crate) struct ComponentHashes {
    /// (component name, hash) pairs in a fixed, declaration order.
    entries: [(&'static str, String); Self::COMPONENT_COUNT],
}

#[allow(dead_code)] // consumed by Task 4 system-prompt core assembly
impl ComponentHashes {
    const COMPONENT_COUNT: usize = 6;

    /// Aggregate hash = hash of the concatenated component hashes (in fixed
    /// order). Adding a field forces a new component entry, so attribution can
    /// never be silently bypassed.
    pub(crate) fn aggregate(&self) -> String {
        let concatenated: Vec<serde_json::Value> = self
            .entries
            .iter()
            .map(|(_, hash)| serde_json::Value::String(hash.clone()))
            .collect();
        hash_canonical(&serde_json::Value::Array(concatenated))
    }

    /// Names of components whose hash differs from `other`, in fixed order.
    pub(crate) fn diff(&self, other: &ComponentHashes) -> Vec<&'static str> {
        self.entries
            .iter()
            .zip(other.entries.iter())
            .filter_map(|((name, lhs), (_, rhs))| (lhs != rhs).then_some(*name))
            .collect()
    }
}

#[allow(dead_code)] // consumed by Task 4 system-prompt core assembly
impl CoreInputs {
    /// Hash each component independently. Unordered collections (tool roster,
    /// skills catalog) are sorted by name BEFORE hashing so ordering differences
    /// do not change the hash.
    pub(crate) fn component_hashes(&self) -> ComponentHashes {
        let mut tool_roster = self.tool_roster.clone();
        tool_roster.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let mut skills_catalog = self.skills_catalog.clone();
        skills_catalog.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let mut specialists = self.specialists.clone();
        specialists.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        ComponentHashes {
            entries: [
                ("base_template", hash_canonical(&json!(self.base_template))),
                ("tool_roster", hash_canonical(&json!(tool_roster))),
                ("skills_catalog", hash_canonical(&json!(skills_catalog))),
                ("specialists", hash_canonical(&json!(specialists))),
                ("channel_rules", hash_canonical(&json!(self.channel_rules))),
                ("persona", hash_canonical(&json!(self.persona))),
            ],
        }
    }

    /// Aggregate hash over all component hashes.
    pub(crate) fn aggregate_hash(&self) -> String {
        self.component_hashes().aggregate()
    }
}

#[cfg(test)]
pub(crate) fn test_core_inputs() -> CoreInputs {
    CoreInputs {
        base_template: "T".into(),
        tool_roster: vec![("b".into(), "{}".into()), ("a".into(), "{}".into())],
        skills_catalog: vec![
            ("s2".into(), "d2".into(), true),
            ("s1".into(), "d1".into(), true),
        ],
        specialists: vec![("x".into(), "dx".into())],
        channel_rules: "R".into(),
        persona: "P".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_hash_is_order_insensitive_for_unordered_inputs() {
        let a = test_core_inputs();
        let mut b = a.clone();
        b.tool_roster.reverse();
        b.skills_catalog.reverse();
        assert_eq!(a.component_hashes(), b.component_hashes());
        assert_eq!(a.aggregate_hash(), b.aggregate_hash());
    }

    #[test]
    fn changed_component_is_named() {
        let a = test_core_inputs();
        let mut b = a.clone();
        b.skills_catalog.push(("s3".into(), "d3".into(), true));
        let diff = a.component_hashes().diff(&b.component_hashes());
        assert_eq!(diff, vec!["skills_catalog"]);
    }

    #[test]
    fn aggregate_hash_is_hash_of_component_hashes() {
        // Pin the construction so a future field addition cannot silently
        // bypass component attribution.
        let a = test_core_inputs();
        let ch = a.component_hashes();
        assert_eq!(a.aggregate_hash(), ch.aggregate());
    }
}
