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

use aidaemon_events::traits::Message;
use aidaemon_events::TerminalState;
use serde_json::{json, Value};

use crate::prompt::fingerprint::hash_canonical;

#[derive(Clone)]
pub struct CachedRender {
    pub content_fp: String,
    pub renderer_version: u32,
    pub mode_tag: String,  // "archived" | "current"
    pub bytes: Vec<Value>, // the rendered messages
}

/// Canonical fingerprint of the COMPLETE ordered render input + terminal_state.
/// Denylist: embedding, importance, created_at. Message.id retained. Fail-closed.
pub fn content_fp(turn_messages: &[Message], terminal_state: TerminalState) -> String {
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

pub fn render_cache_decision(
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
