# LLM runtime rules

Scoped guide for `src/llm_runtime/`; the repository guide is the root
`CLAUDE.md` (`AGENTS.md` here is a symlink to this file).

## Side Call Rule

A model call outside the agent loop is a side call. Declare it as a
`SideCall` in `crates/aidaemon-runtime/src/llm/side_calls.rs` (purpose, class, model, total
deadline, failure policy) and make it through `SideCallGateway`
(`crates/aidaemon-runtime/src/llm/gateway.rs`): `ask` for a prompt you own, `provider` for
library code that takes a provider. Do not hold a raw provider or record a
call by hand anywhere else; `llm_runtime::door_tests` fails if you do.

A closed question for the optional decision backend (`[decisions]`,
`aidaemon_decisions::Decider`) is a `Decision` in
`crates/aidaemon-runtime/src/llm/decisions.rs`, asked through `SideCallGateway::decide`; it
may skip optional work, pick between equal-authority paths or make a check
stricter, never grant authority or decide completion. One exception, kept
narrow: `UntrustedEgressClear` may clear an egress call after untrusted
content without the full verifier, only on a calibrated P(consistent) of at
least 0.9 from the backend and model it was measured on (Gemini's default
decision model), and it never refuses a call; anything else runs the full
check, and without a decision backend nothing changes. A new decision that
clears a check needs its own measurement first.

A safety judge refuses on any failure. A memory job skips, and never writes
from a tainted run. A side call runs no tools. A request that forces one
named tool keeps that tool, because that is how it asks for structured output:
the call comes back to the caller and nothing executes it. Work that needs
tools to run or a second pass is a kernel agent, not a side call.
