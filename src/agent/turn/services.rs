use super::*;

pub(super) struct AgentServices<'a> {
    pub agent: &'a Agent,
}

impl<'a> AgentServices<'a> {
    pub(super) fn new(agent: &'a Agent) -> Self {
        Self { agent }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn services_view_is_send_sync_when_agent_reference_is() {
        assert_send_sync::<AgentServices<'_>>();
    }
}
