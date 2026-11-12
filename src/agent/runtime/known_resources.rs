//! The context that crosses a handoff lives in `aidaemon-runtime`; the agent
//! reads it from its event log and what it inherited.

pub(crate) use aidaemon_runtime::known_resources::*;

impl super::Agent {
    /// What the work so far has reached, for this turn's context or for a
    /// child it spawns ([`aidaemon_runtime::known_resources::known_resources`]).
    pub(in crate::agent) async fn known_resources(
        &self,
        session_id: &str,
        current_task: Option<&str>,
        projected_turn_ids: &[String],
    ) -> KnownResources {
        known_resources(
            &self.event_store,
            self.inherited_authority.as_ref(),
            session_id,
            current_task,
            projected_turn_ids,
        )
        .await
    }

    /// The `[Handoff context]` section for this agent's turn.
    pub(in crate::agent) async fn handoff_section(
        &self,
        session_id: &str,
        projected_turn_ids: &[String],
    ) -> Option<String> {
        handoff_section(
            &self.event_store,
            self.inherited_authority.as_ref(),
            session_id,
            projected_turn_ids,
        )
        .await
    }
}
