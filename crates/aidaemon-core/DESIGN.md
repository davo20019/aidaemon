# aidaemon-core: design sketch

Status: **experimental**. Not published, not used by the daemon yet. The
point is to argue about the shape of the kernel with code that compiles and
tests that pass, before porting anything from `src/agent/`.

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

About 2,300 lines in `src/`, including doc comments and unit tests, of which
`run_loop.rs` is about 650. For comparison, `src/agent/loop/` in the
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

## Deliberately left out (for now)

- **Streaming.** Will be a second `Provider` method yielding typed deltas,
  with `complete` as a fold over it, and matching `Event` variants.
- **`TypedTool`.** Typed effects, access manifests and replay decisions from
  aidaemon's current `Tool` trait, as an opt-in trait with a blanket adapter.
- **Context strategy.** Compaction/eviction as a `before_model` hook first;
  promote to a dedicated `ContextStrategy` trait only if hooks are not enough.
  Task state that must survive compaction should live outside the transcript.
- **Idempotent replay.** Recovery after a crash redoes the turn in flight;
  tool calls that completed in that turn before the crash are not
  deduplicated. aidaemon's idempotency keys (`execution_id + tool + args
  hash`) and per-tool replay decisions are the model for this.
- **Sub-agents.** `Agent::as_tool()` with explicit context forking and child
  budgets that count toward the parent, in an `agents` crate.
- **OpenTelemetry exporter**, as an `EventSink`.
- **SQLite `RunStore`**, in a `store-sqlite` crate with versioned migrations.

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

## Next steps

1. Review this sketch; change the primitives now, while it is cheap.
2. Add an OpenAI-compatible `Provider` in a `providers` crate and run the
   quickstart against a real model.
3. Add a `core` engine to `harness_eval` and run the existing fixtures
   against it.
4. Port daemon policies as hooks one at a time, keeping only those that show
   a measurable gain in repeated (N-of-M) runs.
