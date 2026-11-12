# aidaemon-agents

**Tier: implementation.** No stability promise while 0.x; may change in any minor version. Depend on it through [`aidaemon-sdk`](https://docs.rs/aidaemon-sdk) unless you need it alone.

Agents as tools: a parent agent delegates bounded tasks to child agents, each in its own session, getting back a condensed result with evidence. Child usage rolls up into the parent's budgets. See the crate docs (`ContextMode`, `ChildBudget`) for construction.

Part of the [aidaemon SDK](https://github.com/davo20019/aidaemon) (shared version 0.1.0). MIT licensed. Docs: https://docs.rs/aidaemon-agents
