# aidaemon-core: design sketch

Status: **experimental**. Not published, not used by the daemon yet. The
kernel and the SDK crates around it are implemented and tested; the daemon
still runs its own loop (see `PLAN.md`, Phase 7).

The reasoning behind each choice comes from the Harness Field Guide, a survey
of 14 harness concepts across Claude Code, Codex, OpenHands, SWE-agent, Aider,
LangGraph, OpenAI Agents SDK, Pydantic AI, Letta, Google ADK, Rig and others,
compared with aidaemon's current implementation.

## The rule

The kernel owns the loop, the tool contract, typed outcomes, receipts,
budgets, stuck detection, events and interrupt/resume. **Every policy is a
hook.** Product features (channels, goals, scheduling, mandates, memory,
specialists) live in other crates and use the public API like any other
caller.

If a daemon feature cannot be built as a hook, add a hook point. Do not add a
kernel primitive.

## Size

About 2,900 lines in `src/`, including doc comments and unit tests, of which
`run_loop.rs` is about 800. For comparison, `src/agent/loop/` in the
daemon is about 62,000 lines. The loop reads top to bottom in one file:
guards, build request, `before_model` hooks, model call, then either stop
hooks or the tool batch.

## Primitives

| Primitive | File | What it is |
|---|---|---|
| `Agent` / `AgentBuilder` | `agent.rs` | Configuration as data with a stable id. No per-run state, cheap to clone. |
| `Provider` | `provider.rs` | `complete(&ModelRequest) -> ModelResponse`. Retries are a wrapper (`Retry`), not loop logic. |
| `Tool` | `tool.rs` | `spec` + `call` required. `annotations` default to destructive; `validate` defaults to a `required`-keys check. |
| `ToolContext` | `tool.rs` | Trusted run/session/call ids and a cancel token, passed *beside* the model's arguments. |
| `Hook` | `hook.rs` | `before_model`, `before_tool`, `after_tool`, `on_stop`, `on_stuck`. All default to no-ops. |
| `Receipt` | `receipt.rs` | One per tool call, refusals included: status, stage, duration, reason, structured details. |
| `RunOutcome` / `RunStatus` | `outcome.rs` | Every exit maps to one variant. Usage, turns and receipts travel with it. |
| `RunState` | `state.rs` | Plain serializable data: transcript, receipts, usage, pending interruption. |
| `Event` / `EventSink` | `event.rs` | One ordered stream for UIs, tracing, stores and eval assertions. |
| `RunStore` | `store.rs` | Event journal + latest state. The whole persistence contract of the kernel. |
| `Limits` / `StopCondition` | `limits.rs` | Hard caps (turns clamped to `HARD_TURN_CAP`), plus closure-based custom stop rules. |
| `StuckConfig` | `stuck.rs` | Repeat, repeated-error and ping-pong detection with configurable thresholds. |

## Decisions, and where they come from

**Stop on "no tool calls", plus hard caps that cannot be disabled.** Same as
Claude Agent SDK, Codex, OpenAI Agents SDK, Vercel AI SDK. `max_turns` is
clamped to 1,000 so "unlimited" never means unbounded.

**Typed outcomes.** Following the Claude Agent SDK's result subtypes.
`run()` returns `Ok(RunOutcome)` for everything that happens *inside* a run,
including provider outages and limits; `Err(CoreError)` is reserved for
misuse (resuming another agent's state) and store failures.

**Read-only calls run concurrently, everything else serially, results in
call order.** Claude Agent SDK's rule. The daemon today runs all calls
sequentially (`src/agent/loop/tool_execution/run.rs:826`).

**Errors are tool results, never crashes.** Unknown tool, bad arguments,
denial, timeout, cancellation and tool errors all produce a receipt and a
message the model can react to. The unknown-tool message lists the available
tools.

**Deterministic checks before policy.** Lookup and argument validation happen
before any hook sees the call, so a malformed proposal never reaches a human
approval prompt. Kept from aidaemon's `prepare_invocation` ordering. If a
hook rewrites arguments, they are validated again.

**Trusted context is structural.** aidaemon strips `_`-prefixed fields from
model arguments and re-injects trusted ones (`tool_exec.rs:697-815`). Here the
model's arguments and the harness context are separate parameters, so there
is nothing to strip.

**Receipts for every call.** aidaemon's `ToolReceiptV1`, simplified. No
surveyed harness records refusals as first-class evidence. The receipt is the
basis for completion checks, audits and resume.

**Completion verification is a hook.** `on_stop` can return `Continue` with
a message. The test `stop_hook_can_reject_an_unsupported_done` shows a
receipt-based check rejecting a fabricated "done". aidaemon's assessor,
`CompletionContract` and `RunAggregate` closeout would be one implementation
of this hook, not kernel code; `track_requirements` would be a tool plus an
`on_stop` hook.

**Interrupt and resume across processes.** `before_tool` can return
`Interrupt`; the run ends with `RunStatus::Interrupted` and a serializable
`RunState`. `Agent::resume(state, Approve | Deny)` continues, even in another
process. Approval becomes a pause, not a blocking await on a channel. This is
the LangGraph / OpenAI Agents SDK pattern; no Rust framework has it yet.
State is matched to agents by stable id, avoiding the OpenAI SDK's
name-based rebuild problem.

**Instructions are re-sent, never stored.** The system prompt is prepended on
every request and kept out of `RunState.messages`, so any future compaction
cannot drop it and the cacheable prefix stays byte-stable. Hook edits in
`before_model` affect only the request, not the stored transcript.

**Stuck detection with configurable thresholds.** OpenHands' detector
patterns plus aidaemon's repetition guards, with thresholds in config because
OpenHands' hardcoded ones misfired on polling (their issue #762). The default
policy nudges `max_nudges` times, then stops; an `on_stuck` hook can override
it, for example returning `Ignore` for deliberate polling. A repeated call
with *changing* output is not flagged.

**Output is bounded.** Tool output longer than `max_tool_output_chars` keeps
its head and tail with an explicit notice (SWE-agent's bounded views,
Anthropic's tool guidance).

**Checkpoint every turn.** With a `RunStore`, state is saved after each turn
and at every exit, and every event is journaled. After a crash,
`store.load_state(run_id)` followed by `Agent::run_from` continues from the
last completed turn. The turn in flight is redone, so a non-idempotent tool
in it may run twice (see "Idempotent replay" below).

## Added after the first sketch

- **Streaming.** `Provider::stream` yields `StreamEvent::TextDelta` items and
  one `Done`; the default wraps `complete`. The loop always streams and emits
  `Event::TextDelta`. `Retry` retries a stream only before any output.
- **Usage roll-up.** `ToolOutput::usage` is added to the run, so sub-agent
  spend counts toward the parent's token budget.
- **Access manifests.** `Tool::access` declares reads, writes, hosts and
  commands; `before_tool` hooks receive `ToolInfo { annotations, access }`.
  This replaces the planned separate `TypedTool` trait.
- **Crash-safe batches.** A batch is checkpointed as `InFlight` before tools
  run and after each group. `run_from` on a mid-batch checkpoint finishes the
  batch: unstarted calls run, started idempotent calls re-run, started
  non-idempotent calls get an `Indeterminate` receipt and are not repeated.
- **Typed message origins.** `Message::origin` (`User`, `Harness`,
  `StopHook`) replaces text markers for telling user input from harness
  feedback. Tool results carry `is_error`.
- **Tool timeout hints.** `Tool::timeout` overrides `Limits::tool_timeout`
  for tools with their own timeout argument.
- **Derived agents.** `Agent::with_limits` and `Agent::with_hook` give a
  child agent its own budget or policies without rebuilding it.

## Still left out

- **Async approval inside sub-agents.** A child that interrupts returns an
  error to the parent; resuming a child through its parent is not modelled.
- **Context strategy as a trait.** Compaction is a view-only `before_model`
  hook (`ContextWindow` in `aidaemon-policies`); no dedicated trait yet.
- **Native OpenTelemetry.** `TracingSink` emits `tracing` spans; use a
  `tracing-opentelemetry` layer to export.
- **Reasoning content and images** in `Message`/`ModelResponse`.

## How daemon features would map onto hooks

| Daemon feature today | As a hook |
|---|---|
| Approval broker, risk levels, permission modes | `before_tool` → `Interrupt` / `Deny` |
| Mandate authority gate, correction sandbox | `before_tool` → `Deny` |
| Project-instruction just-in-time discovery | `before_tool` → `Deny` with "new instructions loaded, retry" + `before_model` injection |
| Memory / facts / episodes injection | `before_model` |
| Turn eviction, summarization, render cache | `before_model` |
| Untrusted-output sanitization, secret redaction | `after_tool` |
| Completion contract / ledger closeout / checklist | `on_stop` |
| Approach pivot, self-correction ledger | `on_stuck` |
| Budget auto-extension | custom `StopCondition` + config |
| Decision points, gate telemetry | `EventSink` |

## The SDK crates

| Crate | What it provides |
|---|---|
| `aidaemon-core` | The kernel described above |
| `aidaemon-providers` | OpenAI-compatible and Anthropic providers, streaming, prompt caching |
| `aidaemon-tools` | Workspace-confined file tools, glob/grep, shell over sandbox backends, web fetch, checklist + gate |
| `aidaemon-store-sqlite` | `RunStore` on SQLite, versioned migrations, recovery queries |
| `aidaemon-mcp` | stdio MCP client exposing server tools as `Tool`s |
| `aidaemon-agents` | Agents as tools, child budgets, depth limits |
| `aidaemon-policies` | The daemon's policies as hooks (see table above) |
| `aidaemon-eval` | YAML scenarios, trajectory assertions, N-of-M reports |
| `aidaemon-memory` | Long-term memory ported from the daemon: facts with supersession, episodes, hybrid recall, decay. Store, embedder and scorer are traits, so every part is replaceable |
| `aidaemon-channels` | The `Channel` contract and `ChannelRunner`: access control, one session per conversation, queueing, streamed status edits, approvals as buttons with `/approve` fallback, `/stop` `/new` `/status` |
| `aidaemon-channel-telegram` / `-slack` / `-discord` | Built-in adapters talking to each platform's API directly (Bot API, Socket Mode, Gateway) |
| `aidaemon-sdk` | One dependency re-exporting all of them behind features |

### Extending the SDK

Every extension point is a public trait, so new functionality can live in
any crate, including third-party ones:

| To add | Implement |
|---|---|
| A chat channel | `aidaemon_channels::Channel` (only `name`, `capabilities`, `run`, `send` are required) |
| A model provider | `aidaemon_core::Provider` |
| A tool | `aidaemon_core::Tool` |
| A policy or guardrail | `aidaemon_core::Hook` |
| A memory backend | `aidaemon_memory::{MemoryStore, Embedder, Scorer}` or `aidaemon_policies::Recall` |
| A sandbox | `aidaemon_tools::ExecutionBackend` |
| Run storage | `aidaemon_core::RunStore` |
| Telemetry | `aidaemon_core::EventSink` |

Roadmap and status live in `PLAN.md`.
