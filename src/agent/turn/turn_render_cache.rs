//! The per-turn render cache lives in `aidaemon-runtime`.

pub(crate) use aidaemon_runtime::turn::turn_render_cache::*;

#[cfg(test)]
mod tests {
    use super::super::turn_render::RENDERER_VERSION;
    use super::*;
    use crate::events::TerminalState;
    use crate::traits::Message;
    use chrono::Utc;
    use serde_json::json;

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
