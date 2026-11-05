# aidaemon-core: design sketch

Status: **experimental**, not published. The kernel is the daemon's only
agent loop (`PLAN.md`, Phase 7). The daemon also runs every model call on
`aidaemon-providers` and its owner approvals and outbound secret scan on
`aidaemon-policies`. Most other SDK crates (tools, channels, MCP, store,
agents) are not yet used by the daemon, which still has its own larger
copies; making the daemon a thin composition of the SDK crates is
`PLAN.md`, Phase 8.

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
| `Provider` | `provider.rs` | `complete(&ModelRequest) -> ModelResponse`. Retries are a wrapper (`Retry`), not loop logic; so are failover (`Failover`, `failover.rs`) and remembering an account that cannot answer until its quota resets (`QuotaGuard`, `quota.rs`). |
| `Tool` | `tool.rs` | `spec` + `call` required. `annotations` default to destructive; `validate` defaults to a `required`-keys check; `effects` declares what a call does (`CallEffects`). |
| `ToolContext` | `tool.rs` | Trusted run/session/call ids, a cancel token, and the agent's typed `Extensions` (application values keyed by type, set with `AgentBuilder::tool_extensions`), passed *beside* the model's arguments. |
| `Hook` | `hook.rs` | `before_model`, `before_tool`, `after_tool`, `on_stop`, `on_stuck`. All default to no-ops. |
| `Receipt` | `receipt.rs` | One per tool call, refusals included: status, stage, duration, reason, structured details. |
| `RunOutcome` / `RunStatus` | `outcome.rs` | Every exit maps to one variant. Usage, turns and receipts travel with it. A tool may end the run with `ToolOutput::finish`: once its whole batch has run, the run ends `Finished { outcome, tool }` with the finish's reply as final text (first finish in call order wins; `on_stop` is not consulted, `on_run_end` is). A recorded finish is kept in `RunState::finishing`, so it survives an approval interrupt, `resume` and crash recovery; once its tool has run it also wins over a mid-batch limit or stuck stop, while a cancel drops it (the owner's cancel wins). |
| `RunState` | `state.rs` | Plain serializable data: transcript, receipts, usage, pending interruption. |
| `Event` / `EventSink` | `event.rs` | One ordered stream for UIs, tracing, stores and eval assertions. |
| `RunStore` | `store.rs` | Event journal + latest state. The whole persistence contract of the kernel. |
| `Limits` / `StopCondition` | `limits.rs` | Hard caps (turns clamped to `HARD_TURN_CAP`), plus closure-based custom stop rules. |
| `StuckConfig` | `stuck.rs` | Repeat, repeated-error and ping-pong detection with configurable thresholds. |

## Decisions, and where they come from

**Stop on "no tool calls", plus hard caps that cannot be disabled.** Same as
Claude Agent SDK, Codex, OpenAI Agents SDK, Vercel AI SDK. `max_turns` is
clamped to 1,000 so "unlimited" never means unbounded. Budgets are per run
(a session continued with `with_user_message` starts fresh budgets, while
session-wide `usage` keeps accumulating). `max_tool_calls` and the wall-clock
`timeout` are also checked before every call inside a batch, and the model
call and each tool are bounded by the wall-clock time left, so one turn with
many calls, or one hanging call, cannot overrun them. Calls a limit prevents
get a `Skipped` receipt and a tool result, keeping the transcript well formed.

**Typed outcomes.** Following the Claude Agent SDK's result subtypes.
`run()` returns `Ok(RunOutcome)` for everything that happens *inside* a run,
including provider outages and limits; `Err(CoreError)` is reserved for
misuse (resuming another agent's state) and store failures.

**Read-only calls run concurrently, everything else serially, results in
call order.** Claude Agent SDK's rule. The daemon today runs all calls
sequentially (`src/agent/turn/tool_execution/run.rs`).

**Errors are tool results, never crashes.** Unknown tool, bad arguments,
denial, timeout, cancellation, tool errors and tool panics all produce a
receipt and a message the model can react to. The unknown-tool message lists the available
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
basis for completion checks, audits and resume. Each receipt carries its
`run_id`, since a continued session holds several runs whose turn numbers
restart; completion checks should read `HookContext::run_receipts()` so work
from an earlier run is never credited to this one.

**Completion verification is a hook.** `on_stop` can return `Continue` with
a message. The test `stop_hook_can_reject_an_unsupported_done` shows a
receipt-based check rejecting a fabricated "done". aidaemon's assessor,
`CompletionContract` and `RunAggregate` closeout would be one implementation
of this hook, not kernel code; `track_requirements` would be a tool plus an
`on_stop` hook.

**Interrupt and resume across processes.** `before_tool` can return
`Interrupt`; the run ends with `RunStatus::Interrupted` and a serializable
`RunState`. `Agent::resume(state, Approve | Deny)` continues, even in another
process. The saved call is the one the hooks produced, rewrites included, so
`Approve` runs exactly what the approver was shown. Approval becomes a pause,
not a blocking await on a channel. This is
the LangGraph / OpenAI Agents SDK pattern; no Rust framework has it yet.
State is matched to agents by stable id, avoiding the OpenAI SDK's
name-based rebuild problem.

**Instructions are re-sent, never stored.** The system prompt is prepended on
every request and kept out of `RunState.messages`, so any future compaction
cannot drop it and the cacheable prefix stays byte-stable. Hook edits in
`before_model` affect only the request, not the stored transcript.

**Stuck detection with configurable thresholds.** When the policy decides to
stop, the rest of the batch is skipped, not run. OpenHands' detector
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
- **Untrusted content.** Tools declare who authored their output
  (`Tool::source_trust`) and whether a call acts outside the workspace
  (`ToolAnnotations::egress`); the run records taint in
  `RunState::untrusted_seen`. `UntrustedHook` marks untrusted output as data
  and, in `Enforce` mode, interrupts egress calls made after taint.
  `Agent::run_resolving` settles such interruptions through an
  `InterruptResolver` (a clean-check verifier, an approval UI), approving the
  exact call it was shown or denying it with a reason. `MarkOnly` runs the
  call and leaves `after_untrusted` on its receipt.

## Effects, declines and owner approvals

- **Typed call effects.** `Tool::effects(arguments)` returns `CallEffects`:
  a list of `EffectKey`s (class plus normalized target) and whether the
  list is `complete`. Classes are `Egress` (`host:port`), `Delete`, `Write`
  and `Configuration` (absolute, lexically normalized paths), `Credential`
  and `Process` (names), and `Call` (the kernel's key for a declined
  request that declared no effects). Build keys with `EffectKey::path`,
  `EffectKey::egress` and `EffectKey::named`; path keys overlap when one
  is the other or below it. The default is `CallEffects::none()` for a
  read-only call that is not egress and `CallEffects::unknown()`
  otherwise, so a tool annotated `egress` should declare its egress
  target. Hooks see the effects in `ToolInfo::effects`; they are
  recomputed after a rewrite.
- **Required safeguards.** `AgentBuilder::build` refuses an agent with
  tools unless every `Safeguard::REQUIRED` is covered by a hook's
  `safeguards()` or by `AgentBuilder::attest`: `Authority`, `Scope`,
  `UntrustedOutput`, `Approval` (calls needing the owner's approval are
  put to them as an `InterruptKind::OwnerApproval` interruption) and
  `Egress` (outbound calls never carry secrets).
- **The decline invariant.** When an owner-approval interruption is
  resolved `Deny` (whatever the resolver: a tap, a timeout, a send
  failure), the kernel adds the request's effects to
  `RunState::declined`, or its subject as a `Call` key when it declared
  none (a blank subject binds nothing). From then on `decide()` refuses any call whose effects overlap a
  declined key, with `[OWNER DECLINED]` (`OWNER_DECLINED`), before any
  hook runs and again after every rewrite, in the same batch and in later
  turns. No hook can opt out. A new user message
  (`RunState::with_user_message`) clears the declines.
- **`aidaemon_policies::ApprovalPolicy`** (declares `Approval`): hard
  blocks for catastrophic commands, deny rules, the application's
  `ApprovalRules` (`Allow`, `Deny`, `Ask(request)` or `Defer`), allow
  rules, then the `ApprovalMode`. An ask that the `ApprovalGrants` cover
  (for example `SessionGrants`, keyed by each effect plus the subject) is
  allowed; otherwise it becomes a typed owner-approval interruption
  carrying the call's effects. After a decline in the run, a call whose
  effects are incomplete is asked again even where a rule or grant would
  allow it: an egress or destructive call after any decline, and any call
  that is not read-only after a declined delete, write, configuration or
  call.
- **`aidaemon_policies::EgressGuard`** (declares `Egress`): for a call
  annotated `egress` or declaring an `Egress` effect, scans every string
  and object key in the arguments for the `Redactor`'s secret shapes and
  configured literals, and refuses with `[EGRESS BLOCKED]`
  (`EGRESS_BLOCKED`). No shape matches a host and port, so
  `http://127.0.0.1:8765` passes. Register it before `UntrustedHook` and
  `ApprovalPolicy`, so the owner is never asked about a call it refuses.

## Images

`Image` (base64 bytes or a URL) rides on `Message::images` for user
messages and tool results, and on `ToolOutput::images`, which the loop copies
onto the tool message. `Agent::run` and `RunState::with_user_message` take
`impl Into<Message>`, so text still works and a caller can pass
`Message::user(..).with_images(..)`. Providers own the wire mapping:
Anthropic sends image blocks (inside `tool_result` content for tools);
OpenAI-compatible sends `image_url` parts, and because its tool messages are
text-only, follows a run of tool results with one user message holding their
images. The MCP client attaches `image` content blocks. Whether a model can
see images is the caller's decision; the kernel does not gate it. `Debug` on
`Image` prints the size, not the bytes.

## Still left out

- **Async approval inside sub-agents.** A child that interrupts returns an
  error to the parent; resuming a child through its parent is not modelled.
- **Context strategy as a trait.** Compaction is a view-only `before_model`
  hook (`ContextWindow` in `aidaemon-policies`); no dedicated trait yet.
- **Native OpenTelemetry.** `TracingSink` emits `tracing` spans; use a
  `tracing-opentelemetry` layer to export.
- **Reasoning content** in `Message`/`ModelResponse`. (Images are in: see
  "Images" above.)

## How daemon features would map onto hooks

| Daemon feature today | As a hook |
|---|---|
| Approval broker, risk levels, permission modes | `before_tool` → `Interrupt` / `Deny` (done: `ApprovalPolicy` with the daemon's `AppApprovalRules`) |
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
