# aidaemon-coding-agents

Coding agents an aidaemon agent hands work to: Claude Code, Codex, and any
other program it can launch.

Each agent implements `CodingAgent`: the program it launches, the permission
modes it can run in (checked, every command, full access) and how to launch
it in each. When a mode needs the agent's own approval requests answered,
the launch comes with an `AgentHost` that turns each request into an
`AgentAction` for the embedding agent's `ActionJudge`. This crate decides
nothing itself.

Implementation crate of the aidaemon SDK, no stability promise: use it
through `aidaemon-sdk` (feature `coding-agents`).
