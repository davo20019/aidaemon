use super::memory_scope::{is_low_signal_goal_text, scope_goal_memory_to_project_hints};
use super::types::ConsultantOrchestrationCtx;
use crate::agent::consultant_phase::ConsultantPhaseOutcome;
use crate::agent::*;

impl Agent {
    pub(super) async fn build_goal_feed_forward_context(
        &self,
        session_id: &str,
        goal_user_text: &str,
        recent_messages: &[Value],
        project_hints: &[String],
    ) -> Option<String> {
        let low_signal_without_hints =
            project_hints.is_empty() && is_low_signal_goal_text(goal_user_text);

        let (raw_facts, raw_procedures) = if low_signal_without_hints {
            info!(
                session_id,
                "Skipping goal memory retrieval: low-signal goal text without project hints"
            );
            (Vec::new(), Vec::new())
        } else {
            let memory_query = if project_hints.is_empty() {
                goal_user_text.to_string()
            } else {
                format!(
                    "{goal_user_text}\n\nProject context: {}",
                    project_hints.join(" ")
                )
            };
            (
                self.state
                    .get_relevant_facts(&memory_query, 10)
                    .await
                    .unwrap_or_default(),
                self.state
                    .get_relevant_procedures(&memory_query, 5)
                    .await
                    .unwrap_or_default(),
            )
        };

        let (relevant_facts, relevant_procedures) =
            scope_goal_memory_to_project_hints(raw_facts, raw_procedures, project_hints);

        if !project_hints.is_empty()
            && (relevant_facts.is_empty() || relevant_procedures.is_empty())
        {
            info!(
                session_id,
                project_hints = ?project_hints,
                facts = relevant_facts.len(),
                procedures = relevant_procedures.len(),
                "Scoped goal memory to project hints"
            );
        }

        if relevant_facts.is_empty()
            && relevant_procedures.is_empty()
            && recent_messages.is_empty()
            && project_hints.is_empty()
        {
            return None;
        }

        let ctx = json!({
            "relevant_facts": relevant_facts.iter().map(|f| {
                json!({"category": f.category, "key": f.key, "value": f.value})
            }).collect::<Vec<_>>(),
            "relevant_procedures": relevant_procedures.iter().map(|p| {
                json!({"name": p.name, "trigger": p.trigger_pattern, "steps": p.steps})
            }).collect::<Vec<_>>(),
            "recent_messages": recent_messages,
            "project_hints": project_hints,
            "task_results": [],
        });
        Some(serde_json::to_string(&ctx).unwrap_or_default())
    }

    pub(in crate::agent) async fn run_consultant_orchestration_phase(
        &self,
        ctx: &mut ConsultantOrchestrationCtx<'_>,
    ) -> anyhow::Result<Option<ConsultantPhaseOutcome>> {
        if let Some(outcome) = self.maybe_handle_generic_cancel_request(ctx).await? {
            return Ok(Some(outcome));
        }

        // Orchestration routing (always-on).
        let (complexity, _) = classify_intent_complexity(ctx.user_text, ctx.intent_gate);
        let outcome = self.route_consultant_complexity(ctx, complexity).await?;
        Ok(Some(outcome))
    }
}
