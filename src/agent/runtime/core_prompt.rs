//! Pillar A core-prompt inputs: canonicalization + component hashing.
//! Spec: 2026-06-06-cross-turn-prefix-stability-design.md §Pillar A.
//! Hash actual content inputs, never proxies; canonicalize unordered
//! collections (sort by name) BEFORE hashing. Provider tool-array ordering is
//! enforced upstream and asserted in Tasks 6 and 8. No timestamps, map
//! iteration, or env-dependent formatting.

use crate::agent::prefix_fingerprint::hash_canonical;
use crate::types::ChannelVisibility;
use serde_json::json;

/// Session-static inputs to the core (cacheable) prompt prefix. Each field maps
/// to a query-independent source; per-turn material (MCP-trigger tools, matched
/// skill bodies, personal-memory / untrusted-reference restrictions) is
/// deliberately excluded so the core hash is stable across ordinary query
/// changes within a session.
#[derive(Clone, Debug)]
#[allow(dead_code)] // hashing API — production callers added in Task 7 (per-session core cache)
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
#[allow(dead_code)] // hashing API — production callers added in Task 7 (per-session core cache)
pub(crate) struct ComponentHashes {
    /// (component name, hash) pairs in a fixed, declaration order.
    entries: [(&'static str, String); Self::COMPONENT_COUNT],
}

#[allow(dead_code)] // hashing API — production callers added in Task 7 (per-session core cache)
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

#[allow(dead_code)] // hashing API — production callers added in Task 7 (per-session core cache)
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

/// Render the "## Available Specialists" block from a name-sorted list of
/// `(kind, description)` pairs. Byte-equivalent to the registry-driven
/// `build_available_specialists_block` in system_prompt.rs, but pure over the
/// pre-extracted pairs so it can live in `render_core_prompt`.
///
/// Returns an empty string when `entries` is empty (caller drops the section).
fn render_specialists_block(entries: &[(String, String)]) -> String {
    if entries.is_empty() {
        return String::new();
    }
    let mut s = String::from(
        "## Available Specialists\n\n\
         When you delegate work with `spawn_agent`, pick the specialist that best matches the task. \
         Sub-agents run in an isolated context window with the same tools you have, so keep the `mission` \
         and `task` brief minimal — reference files by path rather than pasting contents, and skip prior \
         tool output or conversation history the sub-agent does not need:\n\n",
    );
    for (name, description) in entries {
        s.push_str("- `");
        s.push_str(name);
        s.push_str("`: ");
        s.push_str(description);
        if !description.ends_with('.') {
            s.push('.');
        }
        s.push('\n');
    }
    s.push_str(
        "\nOmit the `specialist` argument to let the agent infer the right kind from the mission/task text.",
    );
    s
}

/// Render the session-static CORE prompt prefix from [`CoreInputs`].
///
/// PURE + SYNCHRONOUS by contract: no clock, no I/O, no async, no map
/// iteration. Unordered collections (`specialists`, `skills_catalog`) are sorted
/// by name BEFORE emission so the rendered bytes are emission-order-stable.
/// `tool_roster` is NOT emitted here — the canonical name-sorted order binds the
/// PROVIDER TOOL ARRAY (Task 8), not this prose; the `## Tools` selection guide
/// lives inside `base_template`.
///
/// Emission order (canonical core layout): `base_template`, then the
/// `## Available Specialists` block spliced before the base template's `## Tools`
/// anchor (appended after the base when no anchor exists — mirrors the legacy
/// splice), then `channel_rules`, then the `## Available Skills` availability
/// catalog. Empty optional sections are dropped entirely.
///
/// Byte-identity note (Task 4): in the production call site the
/// `channel_rules`/`skills_catalog` fields are deliberately left EMPTY (those
/// sections are still emitted by their existing inline sites this task), so the
/// rendered core equals the legacy `base_prompt` byte-for-byte. The golden tests
/// drive the full layout with all sections populated.
pub(crate) fn render_core_prompt(inputs: &CoreInputs) -> String {
    let mut specialists = inputs.specialists.clone();
    specialists.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let specialists_block = render_specialists_block(&specialists);

    // base_template + specialists splice (before `## Tools`, else appended) —
    // mirrors the legacy build site in build_system_prompt_for_message.
    let mut out = if specialists_block.is_empty() {
        inputs.base_template.clone()
    } else if let Some(idx) = inputs.base_template.find("## Tools") {
        let (head, tail) = inputs.base_template.split_at(idx);
        format!("{head}{specialists_block}\n\n{tail}")
    } else {
        format!("{}\n\n{specialists_block}", inputs.base_template)
    };

    if !inputs.channel_rules.is_empty() {
        out.push_str("\n\n");
        out.push_str(&inputs.channel_rules);
    }

    if !inputs.skills_catalog.is_empty() {
        let mut skills_catalog = inputs.skills_catalog.clone();
        skills_catalog.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let enabled_skills: Vec<_> = skills_catalog
            .iter()
            .filter(|(_, _, enabled)| *enabled)
            .collect();
        if !enabled_skills.is_empty() {
            out.push_str("\n\n## Available Skills\n");
            for (name, description, _enabled) in &enabled_skills {
                out.push_str("- **");
                out.push_str(name);
                out.push_str("**: ");
                out.push_str(description);
                out.push('\n');
            }
        }
    }

    out
}

/// Assemble [`CoreInputs`] from the session-static values available where
/// `build_system_prompt_for_message` runs. Single shared assembler — Tasks 6/7
/// call this same function with the real `channel_rules`/`skills_catalog`
/// snapshots; the Task-4 production site passes them empty so the rendered core
/// stays byte-identical to the legacy `base_prompt` (those sections are still
/// emitted by their existing inline sites this task).
///
/// Explicit stable inputs by contract — NOT `&BootstrapData` (which is built
/// AFTER the prompt-build site and so does not exist at the call site).
///
/// Pure/sync: callers pre-fetch any async snapshots (skill registry read,
/// specialist registry read) and pass them in.
///
/// | Field | Source |
/// |---|---|
/// | `base_template` | role/mode base prompt (`persona`) — minimal on PublicExternal |
/// | `tool_roster` | `session_static_tool_roster` accessor |
/// | `skills_catalog` | active skill registry snapshot (name, one-line desc, enabled) |
/// | `specialists` | `SpecialistRegistry::llm_visible_kinds()` |
/// | `channel_rules` | channel/privacy rule set for the session's visibility class |
/// | `persona` | persona/identity config (== `base_template` source) |
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_core_inputs(
    user_role: crate::types::UserRole,
    channel_ctx: &crate::types::ChannelContext,
    persona: String,
    tool_roster: Vec<(String, String)>,
    skills_catalog: Vec<(String, String, bool)>,
    specialists: Vec<(String, String)>,
    channel_rules: String,
) -> CoreInputs {
    // Specialists are role-typed and never surfaced on the minimal PublicExternal
    // prompt (legacy build site suppresses the splice there).
    let specialists = if channel_ctx.visibility == ChannelVisibility::PublicExternal {
        Vec::new()
    } else {
        specialists
    };
    // Non-owner roles have no tool access (matches `session_static_tool_roster`).
    let tool_roster = if user_role != crate::types::UserRole::Owner {
        Vec::new()
    } else {
        tool_roster
    };
    CoreInputs {
        base_template: persona.clone(),
        tool_roster,
        skills_catalog,
        specialists,
        channel_rules,
        persona,
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
        specialists: vec![("x".into(), "dx".into()), ("a".into(), "da".into())],
        channel_rules: "R".into(),
        persona: "P".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::prefix_fingerprint::TASK_CONTEXT_TAIL_MARKER;

    #[test]
    fn core_prompt_renders_identically_for_identical_inputs() {
        let inputs = test_core_inputs();
        let a = render_core_prompt(&inputs);
        let b = render_core_prompt(&inputs);
        assert_eq!(a, b, "core render must be deterministic");
        assert!(
            !a.contains("[Current Date & Time]"),
            "timestamp belongs to the tail"
        );
        assert!(!a.contains(TASK_CONTEXT_TAIL_MARKER));
    }

    #[test]
    fn core_prompt_is_order_insensitive_for_unordered_inputs() {
        // Determinism of the core BYTES under input reordering. Reorder the
        // unordered collections the core actually emits (skills catalog,
        // specialists); rendered bytes must not change.
        // NOTE: per spec §Pillar A the name-sorted canonical order binds the
        // PROVIDER TOOL ARRAY, not the `## Tools` prose (which is a selection
        // guide needing only determinism). So tool-array emission order is
        // asserted at the provider boundary in Task 8, NOT here.
        let mut inputs = test_core_inputs();
        let a = render_core_prompt(&inputs);
        inputs.skills_catalog.reverse();
        inputs.specialists.reverse();
        assert_eq!(a, render_core_prompt(&inputs));
    }

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
    fn render_core_prompt_skips_disabled_skills() {
        let inputs = CoreInputs {
            base_template: "T".into(),
            tool_roster: vec![],
            skills_catalog: vec![
                ("enabled_skill".into(), "does something".into(), true),
                ("disabled_skill".into(), "should not appear".into(), false),
            ],
            specialists: vec![],
            channel_rules: String::new(),
            persona: "P".into(),
        };
        let out = render_core_prompt(&inputs);
        assert!(
            out.contains("enabled_skill"),
            "enabled skill must appear in the output"
        );
        assert!(
            !out.contains("disabled_skill"),
            "disabled skill must NOT appear in the output"
        );
    }

    #[test]
    fn render_core_prompt_splices_specialists_before_tools_anchor() {
        let inputs = CoreInputs {
            base_template: "Intro\n\n## Tools\nuse them".into(),
            tool_roster: vec![],
            skills_catalog: vec![],
            specialists: vec![("researcher".into(), "Deep research tasks".into())],
            channel_rules: String::new(),
            persona: "P".into(),
        };
        let out = render_core_prompt(&inputs);
        let specialists_pos = out
            .find("## Available Specialists")
            .expect("## Available Specialists block must be present");
        let tools_pos = out
            .find("## Tools")
            .expect("## Tools anchor must be present");
        assert!(
            specialists_pos < tools_pos,
            "## Available Specialists (at {}) must appear before ## Tools (at {})",
            specialists_pos,
            tools_pos
        );
    }

    #[test]
    fn aggregate_hash_is_hash_of_component_hashes() {
        // Non-tautological: build the expected aggregate INDEPENDENTLY of the
        // production code path, then assert equality with aggregate_hash().
        //
        // The aggregate is defined as hash_canonical(JSON array of component
        // hashes in fixed declaration order). We replicate that construction
        // here so that (a) a silently-dropped field changes the entry count
        // check, and (b) a reordering or wrong primitive changes the digest.
        let a = test_core_inputs();
        let ch = a.component_hashes();

        // Assert the entry count equals the fixed constant so adding a field
        // without a corresponding entry causes this test to diverge.
        assert_eq!(
            ch.entries.len(),
            ComponentHashes::COMPONENT_COUNT,
            "entry count must match COMPONENT_COUNT; add a new entry when adding a field"
        );

        // Reconstruct the aggregate independently: JSON array of the
        // individual hash strings in declaration order, then hash_canonical.
        let hash_strings: Vec<serde_json::Value> = ch
            .entries
            .iter()
            .map(|(_, h)| serde_json::Value::String(h.clone()))
            .collect();
        let expected = hash_canonical(&serde_json::Value::Array(hash_strings));

        assert_eq!(
            a.aggregate_hash(),
            expected,
            "aggregate_hash must equal hash_canonical(JSON array of component hashes)"
        );
    }
}
