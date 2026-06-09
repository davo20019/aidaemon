//! Pillar B: per-session/per-turn render cache. Spec §Render cache.
//!
//! Mirrors Pillar A's proven `core_prompts` cache shape and its pure
//! `core_cache_decision` helper. The cache itself is
//! `HashMap<session_id, HashMap<turn_id, CachedRender>>` (the Agent field
//! `turn_renders`); this module owns the cached entry, the canonical
//! fingerprint over a turn's render input, and the PURE cache-decision helper.
//!
//! `content_fp` reuses `prefix_fingerprint::hash_canonical` over the COMPLETE
//! ordered render input (every message in sequence with all rendered fields)
//! PLUS the turn's `terminal_state.tag()`. Denylist (never in the fp):
//! `embedding`, `importance`, `created_at`. `Message.id` IS retained
//! (DB-stable, strengthens input identity; never rendered). Omissions fail
//! CLOSED (a spurious re-render), never open (stale bytes).
//!
//! The debug re-render assertion and the hit/miss/`fp_mismatch` `info!`/`debug!`
//! logging live at the Task-7 call site, NOT in the pure helper — keep the
//! helper pure so it stays unit-testable, exactly as `core_cache_decision`.

// Consumed by the message-build integration in Task 7 (Pillar B). Until then
// the type/field/helpers have no in-crate caller outside this module's own
// tests, so silence dead-code lints (mirrors `turn_render.rs`).
#![allow(dead_code)]

// `Message`, `Value`, and `json` all live in the `agent` module scope.
use super::prefix_fingerprint::hash_canonical;
use super::*;
use crate::events::TerminalState;

#[derive(Clone)]
pub(crate) struct CachedRender {
    pub content_fp: String,
    pub renderer_version: u32,
    pub mode_tag: String,  // "archived" | "current"
    pub bytes: Vec<Value>, // the rendered messages
}

/// Canonical fingerprint of the COMPLETE ordered render input + terminal_state.
/// Denylist: embedding, importance, created_at. Message.id retained. Fail-closed.
pub(crate) fn content_fp(turn_messages: &[Message], terminal_state: TerminalState) -> String {
    let items: Vec<Value> = turn_messages
        .iter()
        .enumerate()
        .map(|(seq, m)| {
            json!({
                "seq": seq,
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "tool_name": m.tool_name,
                "tool_call_id": m.tool_call_id,
                "tool_calls_json": m.tool_calls_json,
                "annotations": m.annotations,
                // EXCLUDED (denylist): embedding, importance, created_at.
            })
        })
        .collect();
    // Stable string tag, NOT `format!("{terminal_state:?}")`: Debug output is not
    // a stability contract, so a variant rename/reorder would silently flip every
    // fp.
    hash_canonical(&json!({ "messages": items, "terminal_state": terminal_state.tag() }))
}

pub(crate) fn render_cache_decision(
    prev: Option<&CachedRender>,
    fp: &str,
    version: u32,
    mode_tag: &str,
    render: impl FnOnce() -> Vec<Value>,
) -> (Vec<Value>, bool, &'static str) {
    if let Some(p) = prev {
        if p.renderer_version != version {
            return (render(), false, "version_mismatch");
        }
        if p.mode_tag != mode_tag {
            return (render(), false, "mode_mismatch");
        }
        if p.content_fp != fp {
            return (render(), false, "fp_mismatch");
        }
        return (p.bytes.clone(), true, "hit");
    }
    (render(), false, "miss")
}

#[cfg(test)]
mod tests {
    use super::super::turn_render::RENDERER_VERSION;
    use super::*;
    use chrono::Utc;

    fn msg(id: &str, role: &str, content: Option<&str>) -> Message {
        Message {
            id: id.to_string(),
            session_id: "sess".to_string(),
            role: role.to_string(),
            content: content.map(|c| c.to_string()),
            tool_call_id: None,
            tool_name: None,
            tool_calls_json: None,
            created_at: Utc::now(),
            annotations: Vec::new(),
            importance: 0.0,
            embedding: None,
            turn_id: Some("turn-1".to_string()),
            attachments: Vec::new(),
        }
    }

    fn sample_turn() -> Vec<Message> {
        let mut tool = msg("m3", "tool", Some("result"));
        tool.tool_name = Some("terminal".to_string());
        tool.tool_call_id = Some("call_1".to_string());
        let mut asst = msg("m2", "assistant", Some("on it"));
        asst.tool_calls_json = Some("[{\"id\":\"call_1\"}]".to_string());
        vec![msg("m1", "user", Some("do the thing")), asst, tool]
    }

    #[test]
    fn fp_stable_for_identical_input() {
        let a = content_fp(&sample_turn(), TerminalState::Completed);
        let b = content_fp(&sample_turn(), TerminalState::Completed);
        assert_eq!(a, b);
    }

    #[test]
    fn fp_changes_when_terminal_state_changes() {
        let t = sample_turn();
        assert_ne!(
            content_fp(&t, TerminalState::Completed),
            content_fp(&t, TerminalState::Failed)
        );
    }

    #[test]
    fn fp_changes_on_late_write() {
        let mut t = sample_turn();
        let base = content_fp(&t, TerminalState::Completed);
        let mut late = msg("m4", "tool", Some("late result"));
        late.tool_name = Some("read_file".to_string());
        late.tool_call_id = Some("call_2".to_string());
        t.push(late);
        assert_ne!(
            content_fp(&t, TerminalState::Completed),
            base,
            "late write must invalidate the turn"
        );
    }

    #[test]
    fn fp_ignores_denylisted_fields() {
        let mut a = sample_turn();
        let fp_a = content_fp(&a, TerminalState::Completed);
        for m in &mut a {
            m.embedding = Some(vec![1.0, 2.0]);
            m.importance = 0.99;
            m.created_at = Utc::now() + chrono::Duration::seconds(7);
        }
        assert_eq!(
            content_fp(&a, TerminalState::Completed),
            fp_a,
            "embedding/importance/created_at excluded"
        );
    }

    #[test]
    fn cache_decision_hit_and_miss() {
        let prev = CachedRender {
            content_fp: "fp1".into(),
            renderer_version: RENDERER_VERSION,
            mode_tag: "archived".into(),
            bytes: vec![json!({"rendered": true})],
        };
        let (_, hit, _) = render_cache_decision(
            Some(&prev),
            "fp1",
            RENDERER_VERSION,
            "archived",
            || unreachable!(),
        );
        assert!(
            hit,
            "matching fp+version+mode is a hit, render_fn NOT called"
        );
        let (_, hit2, reason) =
            render_cache_decision(Some(&prev), "fp2", RENDERER_VERSION, "archived", Vec::new);
        assert!(!hit2);
        assert_eq!(reason, "fp_mismatch");
        let (_, hit3, reason3) = render_cache_decision(
            Some(&prev),
            "fp1",
            RENDERER_VERSION + 1,
            "archived",
            Vec::new,
        );
        assert!(!hit3);
        assert_eq!(reason3, "version_mismatch");
    }

    #[test]
    fn cache_decision_mode_mismatch_and_miss() {
        let prev = CachedRender {
            content_fp: "fp1".into(),
            renderer_version: RENDERER_VERSION,
            mode_tag: "archived".into(),
            bytes: vec![],
        };
        let (_, hit, reason) =
            render_cache_decision(Some(&prev), "fp1", RENDERER_VERSION, "current", Vec::new);
        assert!(!hit);
        assert_eq!(reason, "mode_mismatch");
        let (_, hit2, reason2) =
            render_cache_decision(None, "fp1", RENDERER_VERSION, "archived", Vec::new);
        assert!(!hit2);
        assert_eq!(reason2, "miss");
    }
}
