#[derive(Debug, Default)]
pub(in crate::agent) struct PendingDirectives {
    pending_background_ack: Option<String>,
    pending_external_action_ack: Option<String>,
    pending_system_messages: Vec<crate::agent::SystemDirective>,
    identity_prefill_text: Option<String>,
    cli_agent_boundary_injected: bool,
}

pub(in crate::agent) struct StoppingDirectivesState<'a> {
    pub pending_system_messages: &'a mut Vec<crate::agent::SystemDirective>,
    pub pending_background_ack: &'a mut Option<String>,
}

pub(in crate::agent) struct MessageBuildDirectivesState<'a> {
    pub pending_system_messages: &'a mut Vec<crate::agent::SystemDirective>,
}

pub(in crate::agent) struct LlmDirectivesState<'a> {
    pub pending_system_messages: &'a mut Vec<crate::agent::SystemDirective>,
    pub pending_external_action_ack: &'a mut Option<String>,
    pub identity_prefill_text: &'a mut Option<String>,
}

pub(in crate::agent) struct ResponseDirectivesState<'a> {
    pub pending_system_messages: &'a mut Vec<crate::agent::SystemDirective>,
    pub identity_prefill_text: &'a mut Option<String>,
    pub pending_background_ack: &'a mut Option<String>,
    pub pending_external_action_ack: &'a mut Option<String>,
}

pub(in crate::agent) struct ToolPreludeDirectivesState<'a> {
    pub pending_system_messages: &'a mut Vec<crate::agent::SystemDirective>,
}

pub(in crate::agent) struct ToolExecutionDirectivesState<'a> {
    pub pending_system_messages: &'a mut Vec<crate::agent::SystemDirective>,
    pub cli_agent_boundary_injected: &'a mut bool,
    pub pending_background_ack: &'a mut Option<String>,
    pub pending_external_action_ack: &'a mut Option<String>,
}

impl PendingDirectives {
    pub(in crate::agent) fn for_stopping_phase(&mut self) -> StoppingDirectivesState<'_> {
        StoppingDirectivesState {
            pending_system_messages: &mut self.pending_system_messages,
            pending_background_ack: &mut self.pending_background_ack,
        }
    }

    pub(in crate::agent) fn for_message_build_phase(&mut self) -> MessageBuildDirectivesState<'_> {
        MessageBuildDirectivesState {
            pending_system_messages: &mut self.pending_system_messages,
        }
    }

    pub(in crate::agent) fn for_llm_phase(&mut self) -> LlmDirectivesState<'_> {
        LlmDirectivesState {
            pending_system_messages: &mut self.pending_system_messages,
            pending_external_action_ack: &mut self.pending_external_action_ack,
            identity_prefill_text: &mut self.identity_prefill_text,
        }
    }

    pub(in crate::agent) fn for_response_phase(&mut self) -> ResponseDirectivesState<'_> {
        ResponseDirectivesState {
            pending_system_messages: &mut self.pending_system_messages,
            identity_prefill_text: &mut self.identity_prefill_text,
            pending_background_ack: &mut self.pending_background_ack,
            pending_external_action_ack: &mut self.pending_external_action_ack,
        }
    }

    pub(in crate::agent) fn for_tool_prelude_phase(&mut self) -> ToolPreludeDirectivesState<'_> {
        ToolPreludeDirectivesState {
            pending_system_messages: &mut self.pending_system_messages,
        }
    }

    pub(in crate::agent) fn for_tool_execution_phase(
        &mut self,
    ) -> ToolExecutionDirectivesState<'_> {
        ToolExecutionDirectivesState {
            pending_system_messages: &mut self.pending_system_messages,
            cli_agent_boundary_injected: &mut self.cli_agent_boundary_injected,
            pending_background_ack: &mut self.pending_background_ack,
            pending_external_action_ack: &mut self.pending_external_action_ack,
        }
    }

    pub(in crate::agent) fn push_system_message(
        &mut self,
        directive: crate::agent::SystemDirective,
    ) {
        self.pending_system_messages.push(directive);
    }

    pub(in crate::agent) fn system_message_count(&self) -> usize {
        self.pending_system_messages.len()
    }

    pub(in crate::agent) fn take_system_messages(&mut self) -> Vec<crate::agent::SystemDirective> {
        std::mem::take(&mut self.pending_system_messages)
    }

    pub(in crate::agent) fn pending_background_ack(&self) -> Option<&str> {
        self.pending_background_ack.as_deref()
    }

    pub(in crate::agent) fn set_pending_background_ack(&mut self, ack: impl Into<String>) {
        self.pending_background_ack = Some(ack.into());
    }

    pub(in crate::agent) fn take_pending_background_ack(&mut self) -> Option<String> {
        self.pending_background_ack.take()
    }

    pub(in crate::agent) fn pending_external_action_ack(&self) -> Option<&str> {
        self.pending_external_action_ack.as_deref()
    }

    pub(in crate::agent) fn set_pending_external_action_ack(&mut self, ack: impl Into<String>) {
        self.pending_external_action_ack = Some(ack.into());
    }

    pub(in crate::agent) fn take_pending_external_action_ack(&mut self) -> Option<String> {
        self.pending_external_action_ack.take()
    }

    pub(in crate::agent) fn identity_prefill_text(&self) -> Option<&str> {
        self.identity_prefill_text.as_deref()
    }

    pub(in crate::agent) fn set_identity_prefill_text(&mut self, text: impl Into<String>) {
        self.identity_prefill_text = Some(text.into());
    }

    pub(in crate::agent) fn take_identity_prefill_text(&mut self) -> Option<String> {
        self.identity_prefill_text.take()
    }

    pub(in crate::agent) fn cli_agent_boundary_injected(&self) -> bool {
        self.cli_agent_boundary_injected
    }

    pub(in crate::agent) fn set_cli_agent_boundary_injected(&mut self, injected: bool) {
        self.cli_agent_boundary_injected = injected;
    }
}

#[cfg(test)]
mod tests {
    use super::PendingDirectives;
    use crate::agent::SystemDirective;

    #[test]
    fn default_state_has_no_pending_directives() {
        let state = PendingDirectives::default();

        assert_eq!(state.system_message_count(), 0);
        assert_eq!(state.pending_background_ack(), None);
        assert_eq!(state.pending_external_action_ack(), None);
        assert_eq!(state.identity_prefill_text(), None);
        assert!(!state.cli_agent_boundary_injected());
    }

    #[test]
    fn pushes_and_takes_system_directives() {
        let mut state = PendingDirectives::default();

        state.push_system_message(SystemDirective::RouteFailsafeActive);
        state.push_system_message(SystemDirective::RecoveryModeModelSwitch);

        assert_eq!(state.system_message_count(), 2);
        let taken = state.take_system_messages();
        assert_eq!(taken.len(), 2);
        assert_eq!(state.system_message_count(), 0);
    }

    #[test]
    fn stores_and_takes_acknowledgements() {
        let mut state = PendingDirectives::default();

        state.set_pending_background_ack("background started");
        state.set_pending_external_action_ack("external action complete");

        assert_eq!(state.pending_background_ack(), Some("background started"));
        assert_eq!(
            state.pending_external_action_ack(),
            Some("external action complete")
        );
        assert_eq!(
            state.take_pending_background_ack(),
            Some("background started".to_string())
        );
        assert_eq!(
            state.take_pending_external_action_ack(),
            Some("external action complete".to_string())
        );
        assert_eq!(state.pending_background_ack(), None);
        assert_eq!(state.pending_external_action_ack(), None);
    }

    #[test]
    fn stores_and_takes_identity_prefill() {
        let mut state = PendingDirectives::default();

        state.set_identity_prefill_text("prefill");

        assert_eq!(state.identity_prefill_text(), Some("prefill"));
        assert_eq!(
            state.take_identity_prefill_text(),
            Some("prefill".to_string())
        );
        assert_eq!(state.identity_prefill_text(), None);
    }

    #[test]
    fn tracks_cli_agent_boundary_injection() {
        let mut state = PendingDirectives::default();

        state.set_cli_agent_boundary_injected(true);
        assert!(state.cli_agent_boundary_injected());

        state.set_cli_agent_boundary_injected(false);
        assert!(!state.cli_agent_boundary_injected());
    }
}
